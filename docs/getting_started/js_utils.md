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
                                                              
### Camera Utilities
This module has a `Camera` object that can be initialized and used easily, like below.
```js
// Our input frames will come from here.
const videoElement =
    document.getElementsByClassName('input_video')[0];
const camera = new Camera(videoElement, {
  onFrame: async () => {
    // Send it to a demo, for example, hands
    await hands.send({image: videoElement});
  },
  // Set the width of camera to be rendered
  width: 1280,
  // And set the height
  height: 720
});
// Start the camera.
// Note: It will ask the user for permission, and will error out if this is not given.
camera.start();
```
### Control Panel Utilities
To showcase and monitor the model in elegant ways, MediaPipe has this Control Panel module.
This panel is also convenient to use and set up.                       
One of the most interesting features is the built-in `FPS` object that allows us to monitor the FPS or Frames per Second.
> Note that the event handler function mentioned in the code is [referred here][H-pg]
```js
// Initialize the FPS control
const fpsControl = new FPS();
// In the event handler function onResults() 
function onResults(results) {
    // Tick the FPS, i.e., set Incrementation breaks
    fpsControl.tick();
    // then do the required processing on the results object
    
// Pass in a <div>
new ControlPanel(controlsElement, {
      // Whether to invert the camera, defaults to false
      selfieMode: true,
      // Maximum Number of Hands to detect, a placeholder, only when using Hands API
      maxNumHands: 2,
      // Minimum Confidence score
      minDetectionConfidence: 0.5,
      // Minimum Tracking score
      minTrackingConfidence: 0.5
    })
    .add([
      // a StaticText is simply a label
      new StaticText({title: 'MediaPipe Hands'}),
      // Add the FPS control to the Control Panel
      fpsControl,
      // a Toggle is one with only 2 options (i.e., true or false)
      new Toggle({title: 'Selfie Mode', field: 'selfieMode'}),
      // A slider can have multiple options. This will have the options 1,2,3 and 4. 
      // Range determines maximum and minimum allowed value
      // Step determines the difference between two options
      new Slider(
          {title: 'Max Number of Hands', field: 'maxNumHands', range: [1, 4], step: 1}),
      new Slider({
        title: 'Min Detection Confidence',
        field: 'minDetectionConfidence',
        range: [0, 1],
        step: 0.01
      }),
      new Slider({
        title: 'Min Tracking Confidence',
        field: 'minTrackingConfidence',
        range: [0, 1],
        step: 0.01
      }),
    ])
    // This is run when an option is updated. 
    // the options object will contain parameters for all the above in a array.
    .on(options => {
      videoElement.classList.toggle('selfie', options.selfieMode);
      hands.setOptions(options);
    });

```

[Ho-pg]: ../solutions/holistic.md#javascript-solution-api
[F-pg]: ../solutions/face_mesh.md#javascript-solution-api
[H-pg]: ../solutions/hands.md#javascript-solution-api
[P-pg]: ../solutions/pose.md#javascript-solution-api
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
