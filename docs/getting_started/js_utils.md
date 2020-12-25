---
layout: default
title: JavaScript Utilities
parent: MediaPipe in JavaScript
nav_order: 1
---

# JavaScript Utils for MediaPipe
{: .no_toc }

1. TOC
{:toc}
---

## Utilities
MediaPipe offers utilities for enabling users to use the MediaPipe Examples with ease, and conveniently integrate with existing code.                       
The utilities MediaPipe currently has are ~
* [Drawing Utilities][draw-npm],
* [Camera Utilities][cam-npm], and
* [Control Panel][ctrl-npm] Utilities.

### Drawing Utilities
These are for drawing on canvas using output from MediaPipe's solutions.                 
Currently, this includes three functions ~
1. *drawLandmarks*                                               
```js
drawLandmarks(canvasContext, landmarks, {
            fillColor: fillColor,
            color: color,
            lineWidth: lineWidth
        }
```                  
This function is for drawing landmarks.                                               
The value for the `landmarks` parameter can be found using in the `results` array.

2. *drawConnectors*
```js
drawConnectors(canvasContext, landmarks, LANDMARK_CONSTANTS, {
    color: color,
    lineWidth: lineWidth
})
```                        
This is a function used for plotting connectors in various MediaPipe Solutions.                         
The `LANDMARK_CONSTANTS` are specific for every solution, so you could choose these there.                           

3. *drawRectangle*
```js
drawRectangle(canvasContext, {
    width: width,
    height: height,
    xCenter: xCenter,
    yCenter: yCenter,
    rotation: rotation
}, {
    color: color
})
```
This allows drawing of a rectangle, with a certain width, height and rotation given the coordinates of its location.

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
[H-pg]: ../solutions/hands#javascript-solution-api
[P-pg]: ../solutions/pose#javascript-solution-api
[Ho-npm]: https://www.npmjs.com/package/@mediapipe/holistic
[F-npm]: https://www.npmjs.com/package/@mediapipe/face_mesh
[H-npm]: https://www.npmjs.com/package/@mediapipe/hands
[P-npm]: https://www.npmjs.com/package/@mediapipe/pose
[draw-npm]: https://www.npmjs.com/package/@mediapipe/pose
[cam-npm]: https://www.npmjs.com/package/@mediapipe/pose
[ctrl-npm]: https://www.npmjs.com/package/@mediapipe/pose
[Ho-jsd]: https://www.jsdelivr.com/package/npm/@mediapipe/holistic
[F-jsd]: https://www.jsdelivr.com/package/npm/@mediapipe/face_mesh
[H-jsd]: https://www.jsdelivr.com/package/npm/@mediapipe/hands
[P-jsd]: https://www.jsdelivr.com/package/npm/@mediapipe/pose
[Ho-pen]: https://code.mediapipe.dev/codepen/holistic
[F-pen]: https://code.mediapipe.dev/codepen/face_mesh
[H-pen]: https://code.mediapipe.dev/codepen/hands
[P-pen]: https://code.mediapipe.dev/codepen/pose
[Ho-demo]: https://mediapipe.dev/demo/holistic
[F-demo]: https://mediapipe.dev/demo/face_mesh
[H-demo]: https://mediapipe.dev/demo/hands
[P-demo]: https://mediapipe.dev/demo/pose
[npm]: https://www.npmjs.com/package/@mediapipe
[codepen]: https://code.mediapipe.dev/codepen
