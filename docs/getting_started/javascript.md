---
layout: default
title: MediaPipe in JavaScript
parent: Getting Started
nav_order: 4
---

# MediaPipe in JavaScript
{: .no_toc }

1. TOC
{:toc}
---

## Ready-to-use JavaScript Solutions

MediaPipe currently offers the following solutions:

Solution          | NPM Package                   | Example
----------------- | ----------------------------- | -------
[Face Mesh][F-pg] | [@mediapipe/face_mesh][F-npm] | [mediapipe.dev/demo/face_mesh][F-demo]
[Face Detection][Fd-pg] | [@mediapipe/face_detection][Fd-npm] | [mediapipe.dev/demo/face_detection][Fd-demo]
[Hands][H-pg]     | [@mediapipe/hands][H-npm]     | [mediapipe.dev/demo/hands][H-demo]
[Holistic][Ho-pg] | [@mediapipe/holistic][Ho-npm] | [mediapipe.dev/demo/holistic][Ho-demo]
[Pose][P-pg]      | [@mediapipe/pose][P-npm]      | [mediapipe.dev/demo/pose][P-demo]

Click on a solution link above for more information, including API and code
snippets.

The quickest way to get acclimated is to look at the examples above. Each demo
has a link to a [CodePen][codepen] so that you can edit the code and try it
yourself. We have included a number of utility packages to help you get started:

*   [@mediapipe/drawing_utils][draw-npm] - Utilities to draw landmarks and
    connectors.
*   [@mediapipe/camera_utils][cam-npm] - Utilities to operate the camera.
*   [@mediapipe/control_utils][ctrl-npm] - Utilities to show sliders and FPS
    widgets.

Note: See these demos and more at [MediaPipe on CodePen][codepen]

All of these solutions are staged in [NPM][npm]. You can install any package
locally with `npm install`. Example:

```
npm install @mediapipe/holistic.
```

If you would rather not stage these locally, you can rely on a CDN (e.g.,
[jsDelivr](https://www.jsdelivr.com/)). This will allow you to add scripts
directly to your HTML:

```
<head>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.1/drawing_utils.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.1/holistic.js" crossorigin="anonymous"></script>
</head>
```

Note: You can specify version numbers to both NPM and jsdelivr. They are
structured as `<major>.<minor>.<build>`. To prevent breaking changes from
affecting your work, restrict your request to a `<minor>` number. e.g.,
`@mediapipe/holistic@0.1`.

[Ho-pg]: ../solutions/holistic#javascript-solution-api
[F-pg]: ../solutions/face_mesh#javascript-solution-api
[Fd-pg]: ../solutions/face_detection#javascript-solution-api
[H-pg]: ../solutions/hands#javascript-solution-api
[P-pg]: ../solutions/pose#javascript-solution-api
[Ho-npm]: https://www.npmjs.com/package/@mediapipe/holistic
[F-npm]: https://www.npmjs.com/package/@mediapipe/face_mesh
[Fd-npm]: https://www.npmjs.com/package/@mediapipe/face_detection
[H-npm]: https://www.npmjs.com/package/@mediapipe/hands
[P-npm]: https://www.npmjs.com/package/@mediapipe/pose
[draw-npm]: https://www.npmjs.com/package/@mediapipe/pose
[cam-npm]: https://www.npmjs.com/package/@mediapipe/pose
[ctrl-npm]: https://www.npmjs.com/package/@mediapipe/pose
[Ho-jsd]: https://www.jsdelivr.com/package/npm/@mediapipe/holistic
[F-jsd]: https://www.jsdelivr.com/package/npm/@mediapipe/face_mesh
[Fd-jsd]: https://www.jsdelivr.com/package/npm/@mediapipe/face_detection
[H-jsd]: https://www.jsdelivr.com/package/npm/@mediapipe/hands
[P-jsd]: https://www.jsdelivr.com/package/npm/@mediapipe/pose
[Ho-pen]: https://code.mediapipe.dev/codepen/holistic
[F-pen]: https://code.mediapipe.dev/codepen/face_mesh
[Fd-pen]: https://code.mediapipe.dev/codepen/face_detection
[H-pen]: https://code.mediapipe.dev/codepen/hands
[P-pen]: https://code.mediapipe.dev/codepen/pose
[Ho-demo]: https://mediapipe.dev/demo/holistic
[F-demo]: https://mediapipe.dev/demo/face_mesh
[Fd-demo]: https://mediapipe.dev/demo/face_detection
[H-demo]: https://mediapipe.dev/demo/hands
[P-demo]: https://mediapipe.dev/demo/pose
[npm]: https://www.npmjs.com/package/@mediapipe
[codepen]: https://code.mediapipe.dev/codepen
