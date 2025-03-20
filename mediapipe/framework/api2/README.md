# New MediaPipe APIs

This directory defines new APIs for MediaPipe:

- Node API, an update to the Calculator API for defining MediaPipe components.
- Builder API, for assembling CalculatorGraphConfigs with C++, as an alternative
  to using the proto API directly.

The new APIs interoperate fully with the existing framework code, and we are
adopting them in our calculators. We are still making improvements, and the
placement of this code under the `mediapipe::api2` namespace is not final.

Developers are welcome to try out these APIs as early adopters, but there may be
breaking changes.

## Node API

This API can be used to define calculators. It is designed to be more type-safe
and less verbose than the original API.

Input/output ports (streams and side packets) can now be declared as typed
constants, instead of using plain strings for access.

For example, instead of

```
constexpr char kSelectTag[] = "SELECT";
if (cc->Inputs().HasTag(kSelectTag)) {
  cc->Inputs().Tag(kSelectTag).Set<int>();
}
```

you can write

```
static constexpr Input<int>::Optional kSelect{"SELECT"};
```

Instead of setting up the contract procedurally in `GetContract`, add ports to
the contract declaratively, as follows:

```
MEDIAPIPE_NODE_CONTRACT(kInput, kOutput);
```

To access an input in Process, instead of

```
int select = cc->Inputs().Tag(kSelectTag).Get<int>();
```

write

```
int select = kSelect(cc).Get();  // alternative: *kSelect(cc)
```

Sets of multiple ports can be declared with `::Multiple`. Note, also, that a tag
string must always be provided when declaring a port; use `""` for untagged
ports. For example:


```
for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
  cc->Inputs().Index(i).SetAny();
}
```

becomes

```
static constexpr Input<AnyType>::Multiple kIn{""};
```

For output ports, the payload can be passed directly to the `Send` method. For
example, instead of

```
cc->Outputs().Index(0).Add(
    new std::pair<Packet, Packet>(cc->Inputs().Index(0).Value(),
                                  cc->Inputs().Index(1).Value()),
    cc->InputTimestamp());
```

you can write

```
kPair(cc).Send({kIn(cc)[0].packet(), kIn(cc)[1].packet()});
```

The input timestamp is propagated to the outputs by default. If your calculator
wants to alter timestamps, it must add a `TimestampChange` entry to its contract
declaration. For example:

```
MEDIAPIPE_NODE_CONTRACT(kMain, kLoop, kPrevLoop,
                        StreamHandler("ImmediateInputStreamHandler"),
                        TimestampChange::Arbitrary());
```

Several calculators in
[`calculators/core`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/calculators/core) and
[`calculators/tensor`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/calculators/tensor)
have been updated to use this API. Reference them for more examples.

More complete documentation will be provided in the future.

## Builder API

Documentation will be provided in the future.
