import os
from typing import Optional
from supabase import create_client
from datetime import datetime

def elo_update(elo1, elo2, result, k=32):
    """
    Update two ELO scores based on result.
    :param elo1: Player 1 ELO
    :param elo2: Player 2 ELO
    :param result: 1 if player 1 wins, 0 if player 1 loses, 0.5 for draw
    :param k: K-factor (default 32)
    :return: (new_elo1, new_elo2)
    """
    # Calculate expected scores
    expected1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    expected2 = 1 / (1 + 10 ** ((elo1 - elo2) / 400))
    # Update ratings
    new_elo1 = elo1 + k * (result - expected1)
    new_elo2 = elo2 + k * ((1 - result) - expected2)
    return new_elo1, new_elo2


def get_participant_elo(username: str) -> int:
    """Read ELO for a participant from the ai2_leaderboard table."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    if not key:
        raise RuntimeError("Missing Supabase key in environment (SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY)")

    client = create_client(url, key)

    resp = client.table("ai2_leaderboard").select("elo").eq("username", username).single().execute()
    if getattr(resp, "data", None) and "elo" in resp.data:
        return resp.data["elo"]
    raise ValueError(f"No ELO found for user: {username}")

def update_participant_elo(username: str, elo: int, match_result: Optional[str] = None) -> None:
    """Update ELO for a participant in the ai2_leaderboard table.

    match_result is accepted for compatibility but not used here.
    """
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)
    resp = client.table("ai2_leaderboard").update({"elo": elo}).eq("username", username).execute()

    if hasattr(resp, "error") and resp.error:
        # supabase-py may return error object/message
        raise RuntimeError(str(resp.error))
    
def upload_video_to_supabase(video_path, agent1_username, agent2_username):
    """
    Uploads the video at video_path to the 'battle-videos' bucket in Supabase Storage,
    under the folder '{agent1_username}_vs_{agent2_username}/battle.mp4'.
    Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env variables set.
    """
    import os

    try:
        from supabase import create_client, Client
    except ImportError:
        raise ImportError("You must install supabase-py to use this feature: pip install supabase")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment variables.")

    client: Client = create_client(supabase_url, supabase_key)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bucket_name = "AI2-battles"
    dest_path = f"{agent1_username}_vs_{agent2_username}_{time}_battle.mp4"

    with open(video_path, "rb") as video_file:
        video_data = video_file.read()
        # Remove existing file if it exists
        try:
            client.storage.from_(bucket_name).remove([dest_path])
        except Exception:
            pass  # Ignore if file does not exist

        response = client.storage.from_(bucket_name).upload(
            dest_path,
            video_data,
            file_options={"content-type": "video/mp4"},
        )
    # INSERT_YOUR_CODE
    public_url = client.storage.from_(bucket_name).get_public_url(dest_path)
    print(f"Video uploaded. Public URL: {public_url}")
    return response
