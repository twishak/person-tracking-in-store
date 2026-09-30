"""
Behavioural tests for EpisodeDetector -- the component that decides how many
interest/interaction events get counted, so its rules are worth pinning down
explicitly. Each test maps to a requirement in the assessment brief.

Run:  python src/test_episode_detector.py
"""

from episodes import UNKNOWN, EpisodeDetector

FPS = 30.0


def feed(detector, signals, start_frame=1):
    """Feed a list of per-frame signals; return every episode that closed."""
    episodes = []
    frame = start_frame
    for sig in signals:
        ep = detector.update(frame, frame / FPS, sig)
        if ep:
            episodes.append(ep)
        frame += 1
    ep = detector.flush(frame, frame / FPS)
    if ep:
        episodes.append(ep)
    return episodes


def test_brief_touch_is_not_counted():
    """Walking past a rack (0.5s) must not count as interest.
    Brief: interest requires attention 'for a sustained period'."""
    d = EpisodeDetector(min_duration_s=2.0, max_gap_s=1.0)
    episodes = feed(d, ["rack_A"] * 15)  # 0.5s
    assert episodes == [], f"expected no episode, got {episodes}"


def test_sustained_engagement_counts_once():
    """10s of continuous engagement is ONE event, not many.
    Brief: 'continuously interacting ... should be counted as one interest
    event, rather than being counted repeatedly'."""
    d = EpisodeDetector(min_duration_s=2.0, max_gap_s=1.0)
    episodes = feed(d, ["rack_A"] * 300)  # 10s
    assert len(episodes) == 1, f"expected 1 episode, got {len(episodes)}"
    assert 9.5 < episodes[0].duration_s < 10.5


def test_brief_occlusion_does_not_split():
    """3s engaged, 0.5s wrist occluded, 3s engaged = ONE event.
    This is the documented UNKNOWN-bridging rule."""
    d = EpisodeDetector(min_duration_s=2.0, max_gap_s=1.0)
    episodes = feed(d, ["rack_A"] * 90 + [UNKNOWN] * 15 + ["rack_A"] * 90)
    assert len(episodes) == 1, f"occlusion split the episode: got {len(episodes)}"
    assert episodes[0].bridged_frames == 15
    assert episodes[0].confidence > 0.9


def test_genuine_return_counts_separately():
    """Engage, leave for 3s, come back = TWO events.
    Brief: 'A customer returning to the same shelf after the previous
    interaction has ended should be counted as a new event'."""
    d = EpisodeDetector(min_duration_s=2.0, max_gap_s=1.0)
    episodes = feed(d, ["rack_A"] * 90 + [None] * 90 + ["rack_A"] * 90)
    assert len(episodes) == 2, f"expected 2 episodes, got {len(episodes)}"


def test_unknown_cannot_start_an_episode():
    """Occlusion alone is never evidence of engagement."""
    d = EpisodeDetector(min_duration_s=2.0, max_gap_s=1.0)
    episodes = feed(d, [UNKNOWN] * 300)
    assert episodes == [], f"UNKNOWN fabricated an episode: {episodes}"


def test_switching_racks_closes_and_opens():
    """Moving from rack A to rack B yields one episode each.
    Brief: resolves the 'region between two shelves' case by attributing to
    whichever rack the wrist is actually observed in."""
    d = EpisodeDetector(min_duration_s=2.0, max_gap_s=1.0)
    episodes = feed(d, ["rack_A"] * 90 + ["rack_B"] * 90)
    assert len(episodes) == 2, f"expected 2 episodes, got {len(episodes)}"
    assert episodes[0].label == "rack_A"
    assert episodes[1].label == "rack_B"


def test_wrist_evidence_is_always_a_sure_event():
    """Confirmed wrist-in-ROI makes an event sure, full stop. A mid-episode
    keypoint dropout is a sensing failure, not counter-evidence -- and one long
    enough to matter would have exceeded max_gap and ended the episode."""
    d = EpisodeDetector(min_duration_s=2.0, max_gap_s=1.0, source="wrist")
    episodes = feed(d, ["rack_A"] * 90 + [UNKNOWN] * 25 + ["rack_A"] * 90)
    assert len(episodes) == 1, "blackout should not split the episode"
    assert episodes[0].longest_unknown_run == 25
    assert episodes[0].status == "sure"


def test_body_only_evidence_is_unsure():
    """Sustained presence in the region without a confirmed reach is
    suggestive but not proof, so it is reported as unsure."""
    d = EpisodeDetector(min_duration_s=4.0, max_gap_s=4.0, source="body")
    episodes = feed(d, ["rack_A"] * 200)
    assert len(episodes) == 1
    assert episodes[0].status == "unsure"


def test_duration_excludes_trailing_tolerance():
    """An episode is timed to its last observed frame, not to the frame where
    the gap tolerance finally expired."""
    d = EpisodeDetector(min_duration_s=2.0, max_gap_s=1.0)
    episodes = feed(d, ["rack_A"] * 90 + [None] * 60)
    assert len(episodes) == 1
    assert episodes[0].duration_s < 3.1, f"tolerance inflated duration: {episodes[0].duration_s}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
