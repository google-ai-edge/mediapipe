# MediaPipe Tasks Decision Package

This package contains the decision tasks for MediaPipe.

## Decision Maker

The MediaPipe Decision Maker evaluates structured decision questions against
input text in a single forward pass of a small on-device model. It supports
boolean predicates, categorical choices, ordinal score rubrics and
multi-question schemas, and returns calibrated probabilities and confidence
metrics.

```javascript
import { DecisionMaker, FilesetResolver } from "@mediapipe/tasks-decision";

const decision = await FilesetResolver.forDecisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-decision@latest/wasm");

const decisionMaker = await DecisionMaker.createFromOptions(decision, {
  baseOptions: {
    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/decision_maker/decision_maker.task"
  }
});

// Boolean predicate
const booleanResult = await decisionMaker.evaluateBoolean(
  "I would like a refund for my last order.",
  { condition: "The user is asking for a refund." }
);
console.log(`Value: ${booleanResult.value}, P(true): ${booleanResult.probabilityTrue}`);

// Categorical choice
const choiceResult = await decisionMaker.evaluateChoice(
  "My package never arrived.",
  { criteria: { shipping: "Delivery problems", billing: "Payment issues" } }
);
console.log(`Selected: ${choiceResult.selectedKey}`);

// Ordinal score
const scoreResult = await decisionMaker.evaluateScore(
  "The support agent was friendly and solved my problem quickly.",
  { rubric: ["Very negative", "Negative", "Neutral", "Positive", "Very positive"] }
);
console.log(`Expected score: ${scoreResult.expectedScore}`);

decisionMaker.close();
```

### Privacy Notice

Last modified: June 5, 2026

When you use MediaPipe Tasks, processing of the input data (e.g. images, video,
text) takes place on device, and MediaPipe does not send that input data to
Google servers. As a result, you can use our MediaPipe Tasks APIs for
processing data that should not leave the device.

MediaPipe Tasks APIs send metrics about the performance and utilization of the
APIs in your app to Google. Google uses this metrics data to measure
performance, usage, debug, maintain and improve the MediaPipe Tasks, as further
described in our [Privacy Policy](https://policies.google.com/privacy).

**You are responsible for obtaining informed consent from your app users about
Google's processing of MediaPipe metrics data as required by applicable law.**
