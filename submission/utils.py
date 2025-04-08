import numpy as np


def get_cpg(timestep, seed=0):
    from flygym.examples.locomotion import CPGNetwork

    phase_biases = np.pi * np.array(
        [
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
        ]
    )
    coupling_weights = (phase_biases > 0) * 10

    return CPGNetwork(
        timestep=timestep,
        intrinsic_freqs=np.ones(6) * 36,
        intrinsic_amps=np.ones(6) * 6,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=np.ones(6) * 20,
        init_phases=None,
        init_magnitudes=None,
        seed=seed,
    )


def step_cpg(cpg_network, preprogrammed_steps, action):
    action = np.ones(2)

    amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
    freqs = np.abs(cpg_network.intrinsic_freqs)
    freqs[:3] *= 1 if action[0] > 0 else -1
    freqs[3:] *= 1 if action[1] > 0 else -1
    cpg_network.intrinsic_amps = amps
    cpg_network.intrinsic_freqs = freqs
    cpg_network.step()

    joints_angles = []
    adhesion_onoff = []

    for i, leg in enumerate(preprogrammed_steps.legs):
        # get target angles from CPGs and apply correction
        my_joints_angles = preprogrammed_steps.get_joint_angles(
            leg,
            cpg_network.curr_phases[i],
            cpg_network.curr_magnitudes[i],
        )
        joints_angles.append(my_joints_angles)

        # get adhesion on/off signal
        my_adhesion_onoff = preprogrammed_steps.get_adhesion_onoff(
            leg, cpg_network.curr_phases[i]
        )

        # No adhesion in stumbling or retracted
        adhesion_onoff.append(my_adhesion_onoff)

    joint_angles = np.array(np.concatenate(joints_angles))
    adhesion_onoff = np.array(adhesion_onoff).astype(int)

    return joint_angles, adhesion_onoff
