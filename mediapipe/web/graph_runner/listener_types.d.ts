/**
 * A listener that receives the contents of a non-empty MediaPipe packet and
 * its timestamp.
 */
export type SimpleListener<T> = (data: T, timestamp: number) => void;

/**
 * A listener that receives the current MediaPipe packet timestamp. This is
 * invoked even for empty packet.
 */
export type EmptyPacketListener = (timestamp: number) => void;

/**
 * A listener that receives a single element of vector-returning output packet.
 * Receives one element at a time (in order). Once all elements are processed,
 * the listener is invoked with `data` set to `unknown` and `done` set to true.
 * Intended for internal usage.
 */
export type VectorListener<T> = (data: T, done: boolean, timestamp: number) =>
    void;

/** A listener that will be invoked with an absl::StatusCode and message. */
export type ErrorListener = (code: number, message: string) => void;
