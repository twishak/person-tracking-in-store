"""Temporal event engine: turns a noisy per-frame signal into countable episodes.

Shared unmodified by Task 2 (shelf engagement) and Task 3 (interaction
sessions) -- both need the same rules: a minimum duration, a gap tolerance, no
double counting of one continuous episode, and a new event after a real return.

Each frame carries one of three values: a label (condition holds), None
(observed not to hold), or UNKNOWN (could not be evaluated, e.g. the wrist is
occluded). UNKNOWN sustains an open episode but can never start one -- a
dropped keypoint is a sensing failure, so it must not manufacture events, but
neither is it evidence of engagement.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Sentinel for "signal could not be evaluated this frame" -- distinct from
# None, which means "evaluated, and the condition does not hold".
UNKNOWN = "__unknown__"


@dataclass
class Episode:
    """One discrete, countable period of sustained engagement."""

    label: str
    start_frame: int
    start_time_s: float
    end_frame: int
    end_time_s: float
    observed_frames: int  # frames where the signal was positively observed
    bridged_frames: int  # frames sustained through UNKNOWN/absence
    longest_unknown_run: int = 0  # longest consecutive stretch with no wrist observed

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    @property
    def confidence(self) -> float:
        """Share of the episode actually observed rather than bridged.
        Surfaced in the overlay so a reviewer can see how much of an episode
        rests on inference instead of observation."""
        total = self.observed_frames + self.bridged_frames
        return self.observed_frames / total if total else 0.0

    # What evidence produced this episode:
    #   "wrist" -- a wrist keypoint was confirmed inside the shelf region.
    #             Direct evidence of reaching for product, so it is asserted
    #             as a sure event regardless of any mid-episode keypoint
    #             dropout (a dropout is a sensing failure, and a dropout long
    #             enough to matter would have exceeded max_gap and simply
    #             ended the episode).
    #   "body"  -- the shopper stood inside the shelf region for a sustained
    #             period but their wrist was never confirmed there. Suggestive
    #             of engagement (people do not loiter inside a fixture by
    #             accident) but not proof, so it is reported as unsure rather
    #             than discarded or counted as fact.
    source: str = "wrist"

    @property
    def status(self) -> str:
        return "sure" if self.source == "wrist" else "unsure"


class EpisodeDetector:
    """Converts a per-frame tri-state signal into discrete episodes.

    Parameters
    ----------
    min_duration_s
        A candidate episode shorter than this is discarded, never counted.
        Implements the brief's "sustained period" requirement and rejects
        someone merely walking past a rack.
    max_gap_s
        How long the signal may lapse before the episode is closed. A later
        return beyond this gap becomes a NEW episode, which is exactly the
        distinction the brief asks us to define between continuous engagement
        and a separate return visit.
    """

    def __init__(self, min_duration_s: float, max_gap_s: float, source: str = "wrist"):
        self.min_duration_s = min_duration_s
        self.max_gap_s = max_gap_s
        self.source = source  # stamped onto emitted episodes; see Episode.source

        self._label: Optional[str] = None
        self._start_frame = 0
        self._start_time = 0.0
        self._last_positive_frame = 0
        self._last_positive_time = 0.0
        self._observed = 0
        self._bridged = 0
        self._current_unknown_run = 0
        self._longest_unknown_run = 0

    def update(
        self, frame_idx: int, timestamp_s: float, signal: Optional[str]
    ) -> Optional[Episode]:
        """Feed one frame. Returns a completed Episode when one closes here."""
        # UNKNOWN: sustain an open episode, but never open one.
        if signal == UNKNOWN:
            if self._label is not None:
                self._bridged += 1
                self._current_unknown_run += 1
                self._longest_unknown_run = max(
                    self._longest_unknown_run, self._current_unknown_run
                )
            return None

        # Any non-UNKNOWN observation ends the current blackout.
        self._current_unknown_run = 0

        # Positive observation matching the open episode -> extend it.
        if signal is not None and signal == self._label:
            self._observed += 1
            self._last_positive_frame = frame_idx
            self._last_positive_time = timestamp_s
            return None

        # Positive observation of a DIFFERENT label -> close the old episode
        # and open a new one immediately (the shopper switched racks).
        if signal is not None and self._label is not None and signal != self._label:
            finished = self._close(frame_idx, timestamp_s)
            self._open(signal, frame_idx, timestamp_s)
            return finished

        # Positive observation with nothing open -> open one.
        if signal is not None and self._label is None:
            self._open(signal, frame_idx, timestamp_s)
            return None

        # signal is None (condition observed absent) with an episode open:
        # tolerate it up to max_gap_s, then close.
        if self._label is not None:
            if timestamp_s - self._last_positive_time > self.max_gap_s:
                return self._close(frame_idx, timestamp_s)
            self._bridged += 1
        return None

    def flush(self, frame_idx: int, timestamp_s: float) -> Optional[Episode]:
        """Close any open episode at end of stream (or when a track is lost)."""
        if self._label is None:
            return None
        return self._close(frame_idx, timestamp_s)

    def _open(self, label: str, frame_idx: int, timestamp_s: float) -> None:
        self._label = label
        self._start_frame = frame_idx
        self._start_time = timestamp_s
        self._last_positive_frame = frame_idx
        self._last_positive_time = timestamp_s
        self._observed = 1
        self._bridged = 0
        self._current_unknown_run = 0
        self._longest_unknown_run = 0

    def _close(self, frame_idx: int, timestamp_s: float) -> Optional[Episode]:
        # The episode ends at its last POSITIVE observation, not at the frame
        # that happened to close it -- trailing bridged frames are tolerance,
        # not engagement, and should not inflate the reported duration.
        episode = Episode(
            label=self._label,
            start_frame=self._start_frame,
            start_time_s=self._start_time,
            end_frame=self._last_positive_frame,
            end_time_s=self._last_positive_time,
            observed_frames=self._observed,
            bridged_frames=self._bridged,
            longest_unknown_run=self._longest_unknown_run,
            source=self.source,
        )
        self._label = None
        self._observed = self._bridged = 0
        self._current_unknown_run = self._longest_unknown_run = 0
        return episode if episode.duration_s >= self.min_duration_s else None

    @property
    def open_episode(self) -> Optional[tuple[str, float]]:
        """(label, start_time) of the in-progress episode, for live overlay."""
        return None if self._label is None else (self._label, self._start_time)
