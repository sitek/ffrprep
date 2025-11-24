import pytest
import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import os

from ffrprep.ffrprep_cli import (
    get_parser,
    parse_baseline,
    parse_ref_channels,
    run_ffrprep
)


class TestGetParser:
    """Test the argument parser creation and configuration."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = get_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert "ffrprep" in parser.description

    def test_required_arguments(self):
        """Test that required arguments are properly defined."""
        parser = get_parser()

        # Test with valid required arguments
        args = parser.parse_args([
            '/path/to/bids',
            '/path/to/output',
            'participant'
        ])

        assert args.bids_dir == Path('/path/to/bids')
        assert args.output_dir == Path('/path/to/output')
        assert args.analysis_level == 'participant'

    def test_optional_arguments_defaults(self):
        """Test default values for optional arguments."""
        parser = get_parser()
        args = parser.parse_args([
            '/path/to/bids',
            '/path/to/output',
            'participant'
        ])

        # Check default values
        assert args.stage == 'both'
        assert args.ref_channels == 'average'
        assert args.high_pass == 1.0
        assert args.low_pass == 40.0
        assert args.baseline == '-0.2,0'
        assert args.tmin == -0.2
        assert args.tmax == 0.6
        assert args.n_procs == 1
        assert args.by_event_type is False
        assert args.skip_bids_validation is False
        assert args.participant_label is None
        assert args.work_dir is None

    def test_analysis_level_choices(self):
        """Test that analysis_level only accepts valid choices."""
        parser = get_parser()

        # Valid choices should work
        args = parser.parse_args(['/bids', '/output', 'participant'])
        assert args.analysis_level == 'participant'

        args = parser.parse_args(['/bids', '/output', 'group'])
        assert args.analysis_level == 'group'

        # Invalid choice should raise SystemExit
        with pytest.raises(SystemExit):
            parser.parse_args(['/bids', '/output', 'invalid'])

    def test_stage_choices(self):
        """Test stage argument choices."""
        parser = get_parser()

        # Test valid stage choices
        for stage in ['preprocessing', 'analysis', 'both']:
            args = parser.parse_args(['/bids', '/output', 'participant',
                                      '--stage', stage])
            assert args.stage == stage

        # Invalid stage should raise SystemExit
        with pytest.raises(SystemExit):
            parser.parse_args(['/bids', '/output', 'participant',
                               '--stage', 'invalid'])

    def test_participant_label_multiple(self):
        """Test multiple participant labels."""
        parser = get_parser()
        args = parser.parse_args([
            '/bids', '/output', 'participant',
            '--participant_label', '01', '02', '03'
        ])
        assert args.participant_label == ['01', '02', '03']

    def test_preprocessing_parameters(self):
        """Test preprocessing parameter parsing."""
        parser = get_parser()

        # Test each parameter individually to avoid parsing issues
        args = parser.parse_args([
            '/bids', '/output', 'participant',
            '--ref_channels', 'Cz'
        ])
        assert args.ref_channels == 'Cz'

        args = parser.parse_args([
            '/bids', '/output', 'participant',
            '--high_pass', '0.5'
        ])
        assert args.high_pass == 0.5

        args = parser.parse_args([
            '/bids', '/output', 'participant',
            '--low_pass', '30.0'
        ])
        assert args.low_pass == 30.0

    def test_analysis_parameters(self):
        """Test analysis parameter parsing."""
        parser = get_parser()
        args = parser.parse_args([
            '/bids', '/output', 'participant',
            '--by_event_type'
        ])
        assert args.by_event_type is True

    def test_general_options(self):
        """Test general option parsing."""
        parser = get_parser()
        args = parser.parse_args([
            '/bids', '/output', 'participant',
            '--skip_bids_validation',
            '--n_procs', '4'
        ])

        assert args.skip_bids_validation is True
        assert args.n_procs == 4

    def test_version_argument(self):
        """Test version argument."""
        parser = get_parser()

        # Version should cause SystemExit with code 0
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(['--version'])
        assert excinfo.value.code == 0


class TestParseBaseline:
    """Test baseline string parsing functionality."""

    def test_parse_baseline_comma_separated(self):
        """Test parsing comma-separated baseline values."""
        result = parse_baseline('-0.2,0.0')
        assert result == (-0.2, 0.0)

        result = parse_baseline('-0.1,0.1')
        assert result == (-0.1, 0.1)

        result = parse_baseline('0,0.5')
        assert result == (0.0, 0.5)

    def test_parse_baseline_single_value(self):
        """Test parsing single baseline value."""
        result = parse_baseline('-0.2')
        assert result == (-0.2, 0.0)

        result = parse_baseline('0.1')
        assert result == (0.1, 0.0)

    def test_parse_baseline_invalid_format(self):
        """Test parsing invalid baseline formats."""
        # Should raise ValueError for invalid float conversion
        with pytest.raises(ValueError):
            parse_baseline('invalid')

        with pytest.raises(ValueError):
            parse_baseline('0.1,invalid')


class TestParseRefChannels:
    """Test reference channel parsing functionality."""

    def test_parse_ref_channels_average(self):
        """Test parsing average reference."""
        result = parse_ref_channels('average')
        assert result is None

        result = parse_ref_channels('AVERAGE')
        assert result is None

        result = parse_ref_channels('Average')
        assert result is None

    def test_parse_ref_channels_single(self):
        """Test parsing single channel reference."""
        result = parse_ref_channels('Cz')
        assert result == 'Cz'

        result = parse_ref_channels('TP9')
        assert result == 'TP9'

    def test_parse_ref_channels_multiple(self):
        """Test parsing multiple channel reference."""
        result = parse_ref_channels('TP9,TP10')
        assert result == ['TP9', 'TP10']

        result = parse_ref_channels('Cz,FCz,CPz')
        assert result == ['Cz', 'FCz', 'CPz']

    def test_parse_ref_channels_empty_string(self):
        """Test parsing empty reference channels."""
        result = parse_ref_channels('')
        assert result == ''


class TestRunFFRPrep:
    """Test the main run_ffrprep function."""

    @patch('ffrprep.ffrprep_cli.get_parser')
    @patch('ffrprep.ffrprep_cli.validate_input_dir')
    @patch('ffrprep.ffrprep_cli.get_participants')
    @patch('ffrprep.ffrprep_cli.setup_derivatives_directories')
    @patch('ffrprep.ffrprep_cli.create_preprocessing_workflow')
    @patch('ffrprep.ffrprep_cli.create_analysis_workflow')
    def test_run_ffrprep_both_stages(
        self, mock_analysis_wf, mock_preproc_wf, mock_setup_dirs,
        mock_get_participants, mock_validate, mock_parser
    ):
        """Test running both preprocessing and analysis stages."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Setup mocks
            mock_args = MagicMock()
            mock_args.bids_dir = Path(tmp_dir) / 'bids'
            mock_args.output_dir = Path(tmp_dir) / 'output'
            mock_args.analysis_level = 'participant'
            mock_args.stage = 'both'
            mock_args.skip_bids_validation = False
            mock_args.participant_label = None
            mock_args.baseline = '-0.2,0'
            mock_args.ref_channels = 'average'
            mock_args.high_pass = 1.0
            mock_args.low_pass = 40.0
            mock_args.tmin = -0.2
            mock_args.tmax = 0.6
            mock_args.by_event_type = False
            mock_args.work_dir = None

            mock_parser.return_value.parse_args.return_value = mock_args
            mock_get_participants.return_value = ['01']

            # Mock derivatives directory setup
            derivatives_root = Path(tmp_dir) / 'derivatives'
            mock_setup_dirs.return_value = {
                'derivatives_root': derivatives_root,
                'preprocessing_dir': (
                    derivatives_root / 'ffrprep-preprocessing'
                ),
                'preprocessing_subject_dir': (
                    derivatives_root / 'ffrprep-preprocessing' / 'sub-01'
                ),
                'analysis_dir': derivatives_root / 'ffrprep-analysis',
                'analysis_subject_dir': (
                    derivatives_root / 'ffrprep-analysis' / 'sub-01'
                )
            }

            # Mock workflows
            mock_preproc_workflow = MagicMock()
            mock_preproc_wf.return_value = mock_preproc_workflow
            mock_analysis_workflow = MagicMock()
            mock_analysis_wf.return_value = mock_analysis_workflow

            # Run the function
            run_ffrprep()

            # Verify calls
            mock_validate.assert_called_once()
            mock_get_participants.assert_called_once_with(
                str(Path(tmp_dir) / 'bids'), None
            )
            mock_setup_dirs.assert_called_once()
            mock_preproc_wf.assert_called_once()
            mock_analysis_wf.assert_called_once()

            # Verify workflow execution
            mock_preproc_workflow.run.assert_called_once()
            mock_analysis_workflow.run.assert_called_once()

    @patch('ffrprep.ffrprep_cli.get_parser')
    @patch('ffrprep.ffrprep_cli.get_participants')
    def test_run_ffrprep_no_participants(
        self, mock_get_participants, mock_parser
    ):
        """Test behavior when no participants are found."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Setup mocks
            mock_args = MagicMock()
            mock_args.bids_dir = Path(tmp_dir) / 'bids'
            mock_args.output_dir = Path(tmp_dir) / 'output'
            mock_args.analysis_level = 'participant'
            mock_args.skip_bids_validation = True
            mock_args.participant_label = ['99']

            mock_parser.return_value.parse_args.return_value = mock_args
            mock_get_participants.return_value = []

            # Run the function
            with patch('builtins.print') as mock_print:
                run_ffrprep()

            # Verify that appropriate message is printed
            mock_print.assert_any_call(
                'No participants found matching [\'99\']'
            )

    @patch('ffrprep.ffrprep_cli.get_parser')
    def test_run_ffrprep_group_level_not_supported(self, mock_parser):
        """Test that group level analysis returns early."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Setup mocks
            mock_args = MagicMock()
            mock_args.bids_dir = Path(tmp_dir) / 'bids'
            mock_args.output_dir = Path(tmp_dir) / 'output'
            mock_args.analysis_level = 'group'

            mock_parser.return_value.parse_args.return_value = mock_args

            # Run the function
            with patch('builtins.print') as mock_print:
                run_ffrprep()

            # Verify that appropriate message is printed
            mock_print.assert_any_call(
                'Currently only participant-level analysis is supported.'
            )

    @patch('ffrprep.ffrprep_cli.get_parser')
    @patch('ffrprep.ffrprep_cli.get_participants')
    @patch('ffrprep.ffrprep_cli.setup_derivatives_directories')
    @patch('ffrprep.ffrprep_cli.check_preprocessing_exists')
    def test_run_ffrprep_analysis_missing_preprocessing(
        self, mock_check_preproc, mock_setup_dirs, mock_get_participants,
        mock_parser
    ):
        """Test analysis-only mode when preprocessing outputs are missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Setup mocks
            mock_args = MagicMock()
            mock_args.bids_dir = Path(tmp_dir) / 'bids'
            mock_args.output_dir = Path(tmp_dir) / 'output'
            mock_args.analysis_level = 'participant'
            mock_args.stage = 'analysis'
            mock_args.skip_bids_validation = True
            mock_args.participant_label = None

            mock_parser.return_value.parse_args.return_value = mock_args
            mock_get_participants.return_value = ['01']
            mock_check_preproc.return_value = (False, [])

            derivatives_root = Path(tmp_dir) / 'derivatives'
            mock_setup_dirs.return_value = {
                'derivatives_root': derivatives_root,
                'analysis_dir': derivatives_root / 'ffrprep-analysis',
                'analysis_subject_dir': (
                    derivatives_root / 'ffrprep-analysis' / 'sub-01'
                )
            }

            # Run the function
            with patch('builtins.print') as mock_print:
                run_ffrprep()

            # Verify error message is printed
            mock_print.assert_any_call(
                'ERROR: No preprocessing outputs found for subject 01.'
            )

    @patch.dict(os.environ, {'IS_DOCKER': '1'})
    @patch('ffrprep.ffrprep_cli.get_parser')
    @patch('ffrprep.ffrprep_cli.validate_input_dir')
    @patch('ffrprep.ffrprep_cli.get_participants')
    @patch('ffrprep.ffrprep_cli.Path')
    def test_run_ffrprep_docker_environment(
        self, mock_path_class, mock_get_participants, mock_validate,
        mock_parser
    ):
        """Test Docker execution environment detection."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Setup mocks
            mock_args = MagicMock()
            mock_args.bids_dir = Path(tmp_dir) / 'bids'
            mock_args.output_dir = Path(tmp_dir) / 'output'
            mock_args.analysis_level = 'participant'
            mock_args.skip_bids_validation = False
            mock_args.participant_label = None

            mock_parser.return_value.parse_args.return_value = mock_args
            mock_get_participants.return_value = []

            # Mock cgroup file to simulate Docker environment
            mock_cgroup_path = MagicMock()
            mock_cgroup_path.exists.return_value = True
            mock_cgroup_path.read_text.return_value = "docker"
            mock_path_class.return_value = mock_cgroup_path

            # Run the function
            run_ffrprep()

            # Verify that validate_input_dir was called with docker exec_env
            mock_validate.assert_called_once_with(
                'docker', Path(tmp_dir) / 'bids', None
            )


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_cli_import(self):
        """Test that CLI module can be imported successfully."""
        # This test verifies that all imports in the CLI module work
        from ffrprep import ffrprep_cli
        assert hasattr(ffrprep_cli, 'get_parser')
        assert hasattr(ffrprep_cli, 'run_ffrprep')
        assert hasattr(ffrprep_cli, 'parse_baseline')
        assert hasattr(ffrprep_cli, 'parse_ref_channels')

    def test_parser_comprehensive(self):
        """Test parser with comprehensive arguments."""
        parser = get_parser()

        # Test with basic required arguments
        args = parser.parse_args([
            '/path/to/bids',
            '/path/to/output',
            'participant'
        ])

        # Verify basic arguments
        assert args.bids_dir == Path('/path/to/bids')
        assert args.output_dir == Path('/path/to/output')
        assert args.analysis_level == 'participant'

        # Test with additional arguments
        args = parser.parse_args([
            '/path/to/bids',
            '/path/to/output',
            'participant',
            '--participant_label', '01', '02',
            '--stage', 'preprocessing',
            '--ref_channels', 'TP9,TP10',
            '--by_event_type',
            '--skip_bids_validation',
            '--n_procs', '8'
        ])

        assert args.participant_label == ['01', '02']
        assert args.stage == 'preprocessing'
        assert args.ref_channels == 'TP9,TP10'
        assert args.by_event_type is True
        assert args.skip_bids_validation is True
        assert args.n_procs == 8
