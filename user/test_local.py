"""
Local validation test - No Supabase connection required
Tests if your SubmittedAgent works properly against a simple opponent
"""
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from loguru import logger
from environment.agent import ConstantAgent, run_match, CameraResolution
from user.my_agent import SubmittedAgent
from user.train_agent import gen_reward_manager

def test_local_agent():
    logger.info("🤖 Initializing your agent...")
    my_agent = SubmittedAgent()
    
    logger.info("🎯 Initializing opponent (ConstantAgent)...")
    opponent = ConstantAgent()
    
    match_time = 90  # 90 seconds
    reward_manager = gen_reward_manager()
    
    logger.info("🥊 Starting validation match...")
    logger.info("   Duration: {} seconds", match_time)
    logger.info("   Note: Skipping video generation (FFmpeg not installed)")
    
    try:
        match_stats = run_match(
            my_agent,
            agent_2=opponent,
            video_path=None,  # Skip video generation
            agent_1_name='Your Rule-Based Agent',
            agent_2_name='ConstantAgent (Does Nothing)',
            resolution=CameraResolution.LOW,
            reward_manager=reward_manager,
            max_timesteps=30 * match_time,
            train_mode=True
        )
        
        logger.success("✅ Validation match completed successfully!")
        logger.info("📊 Match Results:")
        logger.info("   Agent 1 (You) - Damage Dealt: {:.1f}, Damage Taken: {:.1f}, Lives Left: {}", 
                   match_stats.player1.damage_done,
                   match_stats.player1.damage_taken,
                   match_stats.player1.lives_left)
        logger.info("   Agent 2 (Opponent) - Damage Dealt: {:.1f}, Damage Taken: {:.1f}, Lives Left: {}", 
                   match_stats.player2.damage_done,
                   match_stats.player2.damage_taken,
                   match_stats.player2.lives_left)
        logger.info("   Result: {}", match_stats.player1_result)
        logger.success("🎉 Your agent is ready for battle!")
        return True
        
    except Exception as e:
        logger.error("❌ Validation failed: {}", str(e))
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_local_agent()
    exit(0 if success else 1)
