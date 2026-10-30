"""Shared CLI flags; importing this module does not load scientific packages."""


def add_protocol_arguments(parser):
    parser.add_argument(
        '--evaluation-mode', choices=('fixed-current', 'voltage-sweep'),
        default='fixed-current',
        help='CFD protocol: one fixed-current run (default), or the complete voltage sweep.')
    parser.add_argument(
        '--simulation-duration-s', type=float, default=20.,
        help='Fixed-current solver-clock duration (default: 20); not wall-clock seconds. Ignored for voltage-sweep.')
    parser.add_argument(
        '--final-window-s', type=float, default=5.,
        help='Fixed-current final window used for stability checks and objective means (default: 5).')
    parser.add_argument(
        '--target-current-relative-tolerance', type=float, default=.01,
        help='Maximum fixed-current target error for every final-window sample (default: 0.01).')


def protocol_options(args):
    return {key: getattr(args, key) for key in (
        'evaluation_mode', 'simulation_duration_s', 'final_window_s',
        'target_current_relative_tolerance')}
