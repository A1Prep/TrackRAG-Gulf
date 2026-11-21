# Repository alignment

The current `main` branch contains `trackrag/dropoff/` as a package directory and does not contain a competing top-level `trackrag/dropoff.py` module. Keep that layout; do not restore the conflicting file.

All local cases use schema version 1 and preserve their original recording and frame IDs. Coordinates are original-image pixels. Landmark track IDs remain under `source.landmarks`, separate from destination ground-area geometry.

The 50 unchanged approved cases are valid human-reviewed training-contract records. The six corrected candidates intentionally use `review.status=pending_reconfirmation` and must not be loaded as gold until both current reviews are returned.
