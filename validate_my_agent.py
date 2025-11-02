"""
Simple validation script - no dependencies on Supabase or video recording
"""

from environment.agent import ConstantAgent, run_match, CameraResolution
from user.my_agent import SubmittedAgent

print("=" * 60)
print("VALIDATING YOUR TRAINED AGENT")
print("=" * 60)

print("\n1. Loading your trained agent...")
my_agent = SubmittedAgent()
print("   [OK] Agent loaded successfully!")

print("\n2. Running match against ConstantAgent (dummy opponent)...")
print("   (This should be an easy win for your agent)")

opponent = ConstantAgent()

# Run the match WITHOUT video to avoid FFmpeg issues
stats = run_match(
    my_agent,
    agent_2=opponent,
    video_path=None,  # No video
    agent_1_name='Your Agent',
    agent_2_name='Constant Agent',
    resolution=CameraResolution.LOW,
    max_timesteps=30 * 90,  # 90 seconds
    train_mode=False
)

print(f"\n{'=' * 60}")
print(f"MATCH RESULTS")
print(f"{'=' * 60}")
print(f"Your Agent:")
print(f"  - Damage taken: {stats.player1.damage_taken}%")
print(f"  - Lives remaining: {stats.player1.lives_left}")
print(f"  - Damage dealt: {stats.player1.damage_done}")
print(f"\nOpponent (ConstantAgent):")
print(f"  - Damage taken: {stats.player2.damage_taken}%")
print(f"  - Lives remaining: {stats.player2.lives_left}")
print(f"  - Damage dealt: {stats.player2.damage_done}")
print(f"\nMatch time: {stats.match_time:.1f} seconds")
print(f"\nResult: ", end='')

if stats.player1_result.value == 1:
    print("*** YOU WON! Your agent is working! ***")
elif stats.player1_result.value == 0:
    print("DRAW (timeout or tie)")
else:
    print("*** You lost (check the agent) ***")
    
print(f"{'=' * 60}")
print("\nYour agent is ready to submit to the tournament!")
print("Next steps:")
print("1. Push your code to GitHub")
print("2. Run the GitHub Actions validation pipeline")
print("3. Battle other teams!")
print(f"{'=' * 60}")

