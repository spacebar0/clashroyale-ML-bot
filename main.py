"""
Main Entry Point
CLI for training, evaluation, and inference
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import get_logger


def train_imitation(args):
    """Train with imitation learning"""
    logger = get_logger(log_dir="logs")
    logger.info("Starting imitation learning...")
    logger.info(f"Data directory: {args.data}")
    logger.info(f"Epochs: {args.epochs}")
    
    # TODO: Implement imitation learning
    logger.warning("Imitation learning not yet implemented")


def train_ppo(args):
    """Train with PPO self-play"""
    logger = get_logger(log_dir="logs")
    logger.info("Starting PPO training...")
    logger.info(f"Episodes: {args.episodes}")
    
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
    
    # TODO: Implement PPO training
    logger.warning("PPO training not yet implemented")


def evaluate(args):
    """Evaluate trained agent"""
    logger = get_logger(log_dir="logs")
    logger.info("Evaluating agent...")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Episodes: {args.episodes}")
    
    # TODO: Implement evaluation
    logger.warning("Evaluation not yet implemented")


def play(args):
    """Watch agent play live"""
    logger = get_logger(log_dir="logs")
    logger.info("Starting live play...")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # TODO: Implement live play
    logger.warning("Live play not yet implemented")


def test_connection(args):
    """Test ADB connection to BlueStacks"""
    logger = get_logger()
    logger.info("Testing ADB connection...")
    
    from capture import ADBInterface, ScreenCapture
    
    # Test ADB
    adb = ADBInterface(args.device)
    if adb.connect():
        logger.info("✓ ADB connection successful")
        
        resolution = adb.get_screen_resolution()
        logger.info(f"✓ Screen resolution: {resolution}")
        
        screen_on = adb.is_screen_on()
        logger.info(f"✓ Screen on: {screen_on}")
        
        # Test screenshot
        screen = adb.capture_screen()
        if screen is not None:
            logger.info(f"✓ Screenshot captured: {screen.shape}")
        else:
            logger.error("✗ Screenshot failed")
        
        adb.disconnect()
    else:
        logger.error("✗ ADB connection failed")
        logger.error("Make sure BlueStacks is running and ADB is installed")


def test_vision(args):
    """Test vision pipeline"""
    logger = get_logger()
    logger.info("Testing vision pipeline...")
    
    # TODO: Implement vision test
    logger.warning("Vision test not yet implemented")


def test_game_state(args):
    """Test game state extraction"""
    logger = get_logger()
    logger.info("Testing game state...")
    
    from game import CardDatabase, GameState, ActionSpace
    
    # Test card database
    logger.info("\n=== Card Database ===")
    db = CardDatabase()
    logger.info(f"✓ Loaded {len(db.cards)} cards")
    
    # Test game state
    logger.info("\n=== Game State ===")
    state = GameState(
        elixir=7,
        cards_in_hand=['knight', 'archers', 'giant', 'fireball']
    )
    logger.info(f"✓ Created game state: {state}")
    
    # Test action space
    logger.info("\n=== Action Space ===")
    action_space = ActionSpace()
    action = action_space.sample_random()
    logger.info(f"✓ Sampled action: {action}")


def main():
    parser = argparse.ArgumentParser(
        description="Clash Royale Vision-Based RL Agent"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train agent')
    train_parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['imitation', 'ppo'],
        help='Training mode'
    )
    train_parser.add_argument(
        '--data',
        type=str,
        default='data/processed',
        help='Data directory for imitation learning'
    )
    train_parser.add_argument(
        '--checkpoint',
        type=str,
        help='Checkpoint to load'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs (imitation)'
    )
    train_parser.add_argument(
        '--episodes',
        type=int,
        default=10000,
        help='Number of episodes (PPO)'
    )
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate agent')
    eval_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Checkpoint to evaluate'
    )
    eval_parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes'
    )
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Watch agent play live')
    play_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Checkpoint to use'
    )
    
    # Test commands
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument(
        '--component',
        type=str,
        choices=['connection', 'vision', 'game_state'],
        default='connection',
        help='Component to test'
    )
    test_parser.add_argument(
        '--device',
        type=str,
        default='127.0.0.1:5555',
        help='ADB device ID'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        if args.mode == 'imitation':
            train_imitation(args)
        elif args.mode == 'ppo':
            train_ppo(args)
    
    elif args.command == 'eval':
        evaluate(args)
    
    elif args.command == 'play':
        play(args)
    
    elif args.command == 'test':
        if args.component == 'connection':
            test_connection(args)
        elif args.component == 'vision':
            test_vision(args)
        elif args.component == 'game_state':
            test_game_state(args)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
