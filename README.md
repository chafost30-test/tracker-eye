# Tracker Eye

A browser-based object tracker built for a school Maker Faire project.

- Locks onto a target by colour (with round-shape filtering for balls)
- Predicts where it is going: 3D gravity + bounces for balls, turning arcs for cars
- Two floor rulers calibrate the camera so everything is measured in real feet

**Use it:** open the GitHub Pages link for this repo, click **Start camera**, allow camera access, then click the target in the video.

- Learns what the target looks like: an example library (folder or live labelling), an AI double-check (MobileNet), and an Inbox of labels the app suggests
- Records test runs and builds a report zip: scoreboard, filmstrips of the worst moments, frame log, settings
- Replays saved videos to compare versions on the same footage

Code: `index.html` (tracking, physics, layout) and `learn.js` (learning, recording, reports).
