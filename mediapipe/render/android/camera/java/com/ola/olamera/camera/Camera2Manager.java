package com.ola.olamera.camera;


import android.content.Context;
import android.hardware.camera2.CameraManager;
import android.os.Build;
import android.util.Pair;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import com.ola.olamera.camera.camera.CameraLifeManager;
import com.ola.olamera.camera.camera.CameraRepository;
import com.ola.olamera.camera.concurrent.CameraExecutors;
import com.ola.olamera.camera.session.SessionConfig;
import com.ola.olamera.camera.session.UserCameraSession;
import com.ola.olamera.util.Preconditions;

import java.util.Iterator;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executor;

public class Camera2Manager {

    private CameraRepository mCameraRepository;
    private CameraLifeManager mCameraLifeManager;

    private final Object mInitLock = new Object();

    private enum State {
        INITIALED,
        INIITALING,
        UNINITIALED,
    }

    private static State sState = State.UNINITIALED;
    private Context sContext;

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private Camera2Manager(Context context) {
        mCameraRepository = new CameraRepository();
        mCameraLifeManager = new CameraLifeManager(context, mCameraRepository);
        sContext = context;
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public UserCameraSession requestCameraSession(SessionConfig config) {
        return new UserCameraSession(mCameraLifeManager, config);
    }


    public static abstract class InitFeature<T> implements Runnable {


        private T mValue;
        private volatile boolean sHasComplete;


        public synchronized void setComplete(T value) {
            sHasComplete = true;
            mValue = value;
            notifyListener();
        }

        private void notifyListener() {
            Iterator<Pair<Listener<T>, Executor>> it = mListeners.iterator();
            while (it.hasNext()) {
                Pair<Listener<T>, Executor> listenerExecutorMap = it.next();
                Listener<T> listener = listenerExecutorMap.first;
                Executor executor = listenerExecutorMap.second;
                if (executor != null) {
                    executor.execute(() -> listener.onComplete(mValue));
                } else {
                    listener.onComplete(mValue);
                }
                it.remove();
            }
        }

        public synchronized T get() {
            return mValue;
        }


        private ConcurrentLinkedQueue<Pair<Listener<T>, Executor>> mListeners = new ConcurrentLinkedQueue<>();

        public void addListener(@NonNull Listener<T> listener, @NonNull Executor executor) {
            mListeners.add(new Pair<>(listener, executor));
            if (sHasComplete) {
                notifyListener();
            }
        }


    }

    public interface Listener<T> {
        public void onComplete(T result);
    }

    private static Camera2Manager sInstance;

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private InitFeature<Camera2Manager> mCameraInitComplete = new InitFeature<Camera2Manager>() {


        @Override
        public void run() {
            synchronized (mInitLock) {
                if (sState == State.UNINITIALED) {
                    sState = State.INIITALING;
                    CameraExecutors.init();
                    CameraExecutors.getCameraTaskExecutor().execute(() -> {

                        Preconditions.setCameraThread(Thread.currentThread().getId());

                        synchronized (mInitLock) {
                            if (sState == State.INITIALED) {
                                mCameraInitComplete.setComplete(sInstance);
                                return;
                            }
                        }

                        CameraManager cameraManager = (CameraManager) sContext.getSystemService(Context.CAMERA_SERVICE);

                        mCameraRepository.init(cameraManager, CameraExecutors.getCameraTaskExecutor(), CameraExecutors.getCameraTaskExecutor().getHandler());

                        synchronized (mInitLock) {
                            sState = State.INITIALED;
                        }

                        mCameraInitComplete.setComplete(sInstance);
                    });
                } else {
                    mCameraInitComplete.setComplete(sInstance);
                }
            }
        }
    };

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public static synchronized InitFeature<Camera2Manager> getInstance(Context context) {
        if (sInstance == null) {
            sInstance = new Camera2Manager(context);
        }
        sInstance.mCameraInitComplete.run();
        return sInstance.mCameraInitComplete;
    }


    //TODO 目前现在浏览器退出的时候，销毁线程和缓存，后续在相机所有窗口退出的时候销毁
    public synchronized static void unInitIfNeed() {
        if (sInstance != null) {
            sInstance.unInit();
        }
    }


    public void unInit() {
        synchronized (Camera2Manager.class) {
            if (sState == State.UNINITIALED) {
                return;
            }
            CameraExecutors.unInit();
            sState = State.UNINITIALED;
        }
    }
}
