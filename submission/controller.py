from typing import Tuple, Optional, List, Dict
import numpy.typing as npt
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from typing import NamedTuple
import itertools
import numpy as np 




class CommandWithImportance(NamedTuple):
    left_descending_signal: float
    right_descending_signal: float
    importance: float # 0->1


class Controller(BaseController):
    # Class-level constants
    DELTA_MAX = 1.0
    DELTA_MIN = 0.2
    ODOR_GAIN = 2.0
    
    def __init__(self, timestep=1e-4, seed=0):
        if timestep <= 0:
            raise ValueError("timestep must be positive")
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")
            
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.timestep = timestep
        self.time = 0
        
        # Path integration state
        self.current_position = np.array([0.0, 0.0])  # Starting position
        self.current_heading = 0.0
        self.start_position = np.array([0.0, 0.0])
        self.has_reached_odor = False
        
        # Path integration parameters
        self.last_end_effector_pos = None
        self.window_len = int(0.12 / timestep)  # 120ms window
        self.contact_force_thr = 1.0

        # Vision processing parameters - fine-tuned values
        self.vision_threshold = 0.25  # Slightly more sensitive
        self.motion_threshold = 0.15  # More sensitive to motion
        self.visual_memory_window = 5  # Increased for smoother detection
        self.min_patch_size = 8  # Reduced to detect smaller obstacles
        
        # Safety parameters
        self.max_position_change = 1.0  # Maximum position change per step (mm)
        self.drift_warning_distance = 100.0  # Distance for drift warning (mm)
        
        # Initialize vision buffer
        self._vision_buffer = []  # First initialization

        # Initialize missing state variables
        self.prev_vision_data = None
        self._last_behavior = "odor_tracking"
        self._last_vision_result = None
        self._last_vision_time = -1
        self._behavior_timer = 0

    def get_odor_taxis(self, obs: Observation) -> CommandWithImportance:
        try:
            I_right = ((obs["odor_intensity"][0][1] + obs["odor_intensity"][0][3]))/2
            I_left = ((obs["odor_intensity"][0][0] + obs["odor_intensity"][0][2]))/2
            total = (I_left + I_right + 1e-6)/2  # Avoid division by zero
            assymmetry = (I_left - I_right) / total
            s = assymmetry * self.ODOR_GAIN
            # print(s)
            turning_bias = np.tanh(s**2)

            if s > 0: 
                right_descending_signal = self.DELTA_MAX - (self.DELTA_MAX - self.DELTA_MIN) * turning_bias
                left_descending_signal = self.DELTA_MAX
            else: 
                left_descending_signal = self.DELTA_MAX - (self.DELTA_MAX - self.DELTA_MIN) * turning_bias
                right_descending_signal = self.DELTA_MAX

            return CommandWithImportance(left_descending_signal, right_descending_signal, 0.5)
        except Exception as e:
            print(f"Error in odor taxis: {e}")
            return CommandWithImportance(0.5, 0.5, 0.1)  # Safe fallback
    
    def pillar_avoidance(self, obs: Observation, odor_taxis_command: CommandWithImportance) -> CommandWithImportance:
        # Process vision data
        left_eye, right_eye = self.process_vision(obs["vision"])
        
        # Detect obstacles
        left_threat = self.detect_pillars(left_eye)
        right_threat = self.detect_pillars(right_eye)
        
        if left_threat or right_threat:
            # Strong avoidance turn
            if left_threat:
                return CommandWithImportance(0.2, 1.0, 0.9)  # Turn right
            else:
                return CommandWithImportance(1.0, 0.2, 0.9)  # Turn left
        
        return odor_taxis_command

    def ball_avoidance(self, obs: Observation) -> CommandWithImportance:
        # Process current vision
        left_eye, right_eye = self.process_vision(obs["vision"])
        
        if self.prev_vision_data is None:
            self.prev_vision_data = (left_eye, right_eye)
            return CommandWithImportance(0, 0, 0)
        
        prev_left, prev_right = self.prev_vision_data
        
        # Detect motion using temporal difference
        left_motion = np.mean(np.abs(left_eye - prev_left)) > self.motion_threshold
        right_motion = np.mean(np.abs(right_eye - prev_right)) > self.motion_threshold
        
        # Store current vision for next frame
        self.prev_vision_data = (left_eye, right_eye)
        
        # Generate avoidance command
        if left_motion or right_motion:
            if left_motion > right_motion:
                return CommandWithImportance(0.2, 1.0, 1.0)  # Turn right
            else:
                return CommandWithImportance(1.0, 0.2, 1.0)  # Turn left
                
        return CommandWithImportance(0, 0, 0)

    def get_actions(self, obs: Observation) -> Action:
        try:
            self.time = self.time + self.timestep
            self.update_position_estimate(obs)
            
            # Get behavior command
            command = self.get_odor_taxis(obs)
            
            # Get CPG output
            action = np.array([command.left_descending_signal, command.right_descending_signal])
            joint_angles, _ = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.preprogrammed_steps,
                action=action,
            )
            
            # CRITICAL FIX: Always take exactly 44 angles from whatever CPG outputs
            if joint_angles.shape[0] > 44:
                joint_angles = joint_angles[:44]  # Take first 44 angles if we have more
            elif joint_angles.shape[0] < 44:
                joint_angles = np.pad(joint_angles, (0, 44 - joint_angles.shape[0]), 'constant')  # Pad if we have less
                
            # Ensure correct dtype
            joint_angles = joint_angles.astype(np.float32)
            adhesion = np.ones(6, dtype=np.float32)
            
            # Final shape verification
            assert joint_angles.shape == (44,), f"Wrong joint angles shape: {joint_angles.shape}"
            assert adhesion.shape == (6,), f"Wrong adhesion shape: {adhesion.shape}"
            
            return {
                "joints": joint_angles,
                "adhesion": adhesion
            }
            
        except Exception as e:
            print(f"Error in get_actions: {e}")
            # Safe fallback with correct shapes
            return {
                "joints": np.zeros(44, dtype=np.float32),
                "adhesion": np.ones(6, dtype=np.float32)
            }

    def done_level(self, obs: Observation):
        if self.quit:
            return True
            
        # Check if we've reached the odor source
        I_total = np.mean(obs["odor_intensity"])
        if I_total > 0.9 and not self.has_reached_odor:  # Threshold for odor source
            self.has_reached_odor = True
            return False
            
        # If we've already found odor, check if we're back at start
        if self.has_reached_odor:
            dist_to_start = np.linalg.norm(self.current_position - self.start_position)
            if dist_to_start < 2.0:  # 2mm threshold for reaching start
                return True
                
        return False

    def reset(self, **kwargs):
        """Clean reset of all state"""
        self.cpg_network.reset()
        self._vision_buffer = []
        self.prev_vision_data = None
        self._last_behavior = "odor_tracking"
        self._last_vision_result = None
        self._last_vision_time = -1
        self._behavior_timer = 0
        self.current_position = np.array([0.0, 0.0])
        self.current_heading = 0.0

    def update_position_estimate(self, obs: Observation) -> None:
        try:
            # Get end effector positions and handle 3D coordinates
            end_effector_pos = obs["end_effectors"]
            if end_effector_pos.shape != (6, 3):  # Should be (6 legs, 3 coordinates)
                print(f"Unexpected end effector shape: {end_effector_pos.shape}")
                return
                
            # Take only x,y coordinates for position estimation
            end_effector_pos = end_effector_pos[:, :2]  # Take only x,y coordinates
            
            if self.last_end_effector_pos is None:
                self.last_end_effector_pos = end_effector_pos
                return 
                
            # Calculate position change
            end_effector_diff = end_effector_pos - self.last_end_effector_pos
            
            # Update position using mean movement of legs in contact
            contact_forces = obs["contact_forces"]
            contacts = np.linalg.norm(contact_forces, axis=1) > self.contact_force_thr
            if np.any(contacts):
                position_change = np.mean(end_effector_diff[contacts], axis=0)
                self.current_position += position_change
                
            self.last_end_effector_pos = end_effector_pos
            self.current_heading = float(obs["heading"])
            
        except Exception as e:
            print(f"Error in position estimation: {e}")

    def _absolute_to_relative_pos(self, pos: npt.NDArray, base_pos: npt.NDArray, heading: float) -> npt.NDArray:
        rel_pos = pos - base_pos
        angle = heading
        rot_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)]
        ])
        return rel_pos @ rot_matrix.T

    def get_return_command(self) -> CommandWithImportance:
        # Calculate error to start position
        error = self.start_position - self.current_position
        error_dist = np.linalg.norm(error)
        error_heading = np.arctan2(error[1], error[0]) - self.current_heading
        
        # Wrap heading error to [-π, π]
        error_heading = ((error_heading + np.pi) % (2 * np.pi)) - np.pi
        
        # Calculate control signals
        speed_control = np.sqrt(1 / 20 * error_dist)
        speed_heading = 1 * error_heading
        
        left_signal = speed_control * (1 - speed_heading / 2)
        right_signal = speed_control * (1 + speed_heading / 2)
        
        # Clip signals to valid range
        left_signal = np.clip(left_signal, 0.2, 1.0)
        right_signal = np.clip(right_signal, 0.2, 1.0)
        
        return CommandWithImportance(left_signal, right_signal, 0.7)

    def process_vision(self, vision_data: npt.NDArray) -> Tuple[Optional[npt.NDArray], Optional[npt.NDArray]]:
        try:
            # Remove redundant initialization check
            if not hasattr(self, '_vision_buffer'):  # This shouldn't be needed as it's in __init__
                self._vision_buffer = []
            
            # Validate input
            if vision_data is None or vision_data.shape != (2, 721, 2):
                print("Warning: Invalid vision data shape")
                return None, None
                
            # Convert to luminance
            left_eye = np.mean(vision_data[0, :, :], axis=1)
            right_eye = np.mean(vision_data[1, :, :], axis=1)
            
            # Add current frame to buffer
            self._vision_buffer.append((left_eye, right_eye))
            if len(self._vision_buffer) > self.visual_memory_window:
                self._vision_buffer.pop(0)
                
            # Apply temporal smoothing
            left_eye = np.mean([x[0] for x in self._vision_buffer], axis=0)
            right_eye = np.mean([x[1] for x in self._vision_buffer], axis=0)
            
            # Spatial smoothing
            kernel = np.ones(3) / 3
            left_eye = np.convolve(left_eye, kernel, mode='same')
            right_eye = np.convolve(right_eye, kernel, mode='same')
            
            return left_eye, right_eye
        except Exception as e:
            print(f"Error in vision processing: {e}")
            return None, None

    def detect_pillars(self, eye_data: Optional[npt.NDArray]) -> bool:
        """Detect static obstacles"""
        if eye_data is None:
            return False
            
        # Look for dark patches in visual field
        dark_regions = eye_data < self.vision_threshold
        
        # Require minimum patch size to avoid noise
        patch_size = self.min_patch_size  # pixels
        return np.any([sum(1 for _ in g) >= patch_size 
                      for k, g in itertools.groupby(dark_regions) if k])

    def detect_balls(self, eye_data: npt.NDArray, prev_eye_data: npt.NDArray) -> bool:
        # Add validation for array shapes
        if eye_data.shape != prev_eye_data.shape:
            print("Warning: Mismatched eye data shapes")
            return False
            
        # Temporal derivative
        dt = self.timestep
        temporal_diff = (eye_data - prev_eye_data) / dt
        
        # Spatial derivative
        dx = 1  # Angular spacing between ommatidia
        spatial_diff = np.gradient(eye_data, dx)
        
        # HRC detector
        motion_signal = temporal_diff * spatial_diff
        
        # Threshold crossing
        return np.any(np.abs(motion_signal) > self.motion_threshold)

    def get_behavior_priority(self, visual_threat: float, odor_gradient: float, distance_to_goal: float) -> str:
        # Minimum time to maintain behavior
        MIN_BEHAVIOR_TIME = 0.1  # seconds
        
        if self.time - self._behavior_timer < MIN_BEHAVIOR_TIME:
            return self._last_behavior
        
        # Immediate threats override everything
        if visual_threat > 0.8:
            self._last_behavior = "avoidance"
            selected_behavior = "avoidance"
            return selected_behavior  # Add missing return
        
        # Hysteresis for behavior switching
        elif self._last_behavior == "avoidance" and visual_threat > 0.3:
            selected_behavior = "avoidance"
            
        # Navigation priorities
        elif self.has_reached_odor and distance_to_goal < 5.0:
            self._last_behavior = "precise_navigation"
            selected_behavior = "precise_navigation"
            
        else:
            self._last_behavior = "odor_tracking"
            selected_behavior = "odor_tracking"
             
        self._behavior_timer = self.time
        return selected_behavior
        
    def detect_visual_threat(self, obs: Observation) -> float:
        """Compute overall visual threat level with caching"""
        # Cache vision processing results
        if not hasattr(self, '_last_vision_result'):
            self._last_vision_result = None
        if not hasattr(self, '_last_vision_time'):
            self._last_vision_time = -1
            
        # Only process vision if time has changed
        if self.time != self._last_vision_time:
            self._last_vision_result = self.process_vision(obs["vision"])
            self._last_vision_time = self.time
            
        # Add null check for _last_vision_result
        if self._last_vision_result is None:
            return 0.0
        left_eye, right_eye = self._last_vision_result
        # Detect static obstacles (pillars)
        left_pillar = self.detect_pillars(left_eye)
        right_pillar = self.detect_pillars(right_eye)
        
        # Detect moving objects (balls)
        left_ball = right_ball = False
        if self.prev_vision_data is not None:
            prev_left, prev_right = self.prev_vision_data
            left_ball = self.detect_balls(left_eye, prev_left)
            right_ball = self.detect_balls(right_eye, prev_right)
        
        # Update vision memory
        self.prev_vision_data = (left_eye, right_eye)
        
        # Compute threat level (balls have higher priority than pillars)
        threat_level = max(
            float(left_pillar or right_pillar) * 0.9,
            float(left_ball or right_ball)
        )
        
        return threat_level

    def get_joint_angles(self, command: CommandWithImportance) -> np.ndarray:
        """Convert command to joint angles using CPG"""
        try:
            action = np.array([command.left_descending_signal, command.right_descending_signal])
            joint_angles, _ = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.preprogrammed_steps,
                action=action,
            )
            
            # Always ensure we return exactly 44 angles
            if joint_angles.shape[0] == 46:
                return joint_angles[:44].astype(np.float32)  # Take first 44 angles
            elif joint_angles.shape[0] == 42:
                return np.pad(joint_angles, (0, 2), 'constant').astype(np.float32)
            else:
                print(f"Warning: Unexpected joint angles shape: {joint_angles.shape}")
                return np.zeros(44, dtype=np.float32)
                
        except Exception as e:
            print(f"Error in get_joint_angles: {e}")
            return np.zeros(44, dtype=np.float32)

    def get_adhesion_signals(self, command: CommandWithImportance) -> np.ndarray:
        """Generate tripod gait adhesion pattern"""
        adhesion = np.zeros(6)  # Start with all legs up
        phase = (self.time / self.timestep) % 2  # Alternating phases
        
        if phase < 1:
            # Tripod gait: R1, L2, R3 down
            adhesion[::2] = 1
        else:
            # Tripod gait: L1, R2, L3 down
            adhesion[1::2] = 1
        
        return adhesion