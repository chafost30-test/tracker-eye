# Tracker Eye

A browser-based object tracker built for a school Maker Faire project.

- Locks onto a target by colour (with round-shape filtering for balls)
- Predicts where it is going: 3D gravity + bounces for balls, turning arcs for cars
- Two floor rulers calibrate the camera so everything is measured in real feet

**Use it:** open the GitHub Pages link for this repo, click **Start camera**, allow camera access, then click the target in the video.

All the code is in `index.html`. No AI or machine learning: just colour detection, least-squares fitting and physics.
