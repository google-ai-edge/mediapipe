package com.ola.olamera.camera.camera;
/*
 *
 *  Creation    :  20-11-18
 *  Author      : jiaming.wjm@
 */

public enum CameraState {
    /**
     * Camera is waiting for resources to become available before opening.
     *
     * <p>The camera will automatically transition to an {@link #OPENING} state once resources
     * have become available. Resources are typically made available by other cameras closing.
     */
    PENDING_OPEN(/*holdsCameraSlot=*/false),
    /**
     * Camera is in the process of opening.
     *
     * <p>This is a transient state.
     */
    OPENING(/*holdsCameraSlot=*/true),
    /**
     * Camera is open and producing (or ready to produce) image data.
     */
    OPEN(/*holdsCameraSlot=*/true),
    /**
     * Camera is in the process of closing.
     *
     * <p>This is a transient state.
     */
    CLOSING(/*holdsCameraSlot=*/true),
    /**
     * Camera has been closed and should not be producing data.
     */
    CLOSED(/*holdsCameraSlot=*/false),
    /**
     * Camera is in the process of being released and cannot be reopened.
     *
     * <p>This is a transient state. Note that this state holds a camera slot even though the
     * implementation may not actually hold camera resources.
     */
    // TODO: Check if this needs to be split up into multiple RELEASING states to
    //  differentiate between when the camera slot is being held or not.
    RELEASING(/*holdsCameraSlot=*/true),
    /**
     * Camera has been closed and has released all held resources.
     */
    RELEASED(/*holdsCameraSlot=*/false);

    private final boolean mHoldsCameraSlot;

    CameraState(boolean holdsCameraSlot) {
        mHoldsCameraSlot = holdsCameraSlot;
    }

    /**
     * Returns whether a camera in this state could be holding on to a camera slot.
     *
     * <p>Holding on to a camera slot may preclude other cameras from being open. This is
     * generally the case when the camera implementation is in the process of opening a
     * camera, has already opened a camera, or is in the process of closing the camera.
     */
    boolean holdsCameraSlot() {
        return mHoldsCameraSlot;
    }
}
