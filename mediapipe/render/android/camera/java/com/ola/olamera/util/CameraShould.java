package com.ola.olamera.util;

import android.text.TextUtils;
import android.util.Log;

import java.util.Collection;


/**
 * 一个简单的断言工具
 */
public class CameraShould {
    private static final String TAG = "Should";


    private static boolean sShouldThrowError = true;

    /**
     * 设置当Should的条件不满足时是否抛出AssertionError, 如果设置为否，则当Should的条件不满足时，仅
     * 输出error log, 不会抛出AssertionError
     * 一般情况下，开发和内测版本设置抛出异常，正式和灰度版本为了产品的最大可用性，仅输出error log.
     *
     * @param enable 是否需要抛出AssertionError
     */
    public static void setThrowAssertionError(boolean enable) {
        sShouldThrowError = enable;
    }

    public static void notNull(Object obj) {
        notNull(obj, "notNull assert fail");
    }

    public static void notNull(Object obj, String msg) {
        notNullIf(obj, true, msg);
    }

    /**
     * 在ifCondition为true的情况下，obj不应为null
     */
    public static void notNullIf(Object obj, boolean ifCondition) {
        notNullIf(obj, ifCondition, "notNullIf assert fail");
    }

    public static void notNullIf(Object obj, boolean ifCondition, String msg) {
        if (ifCondition && obj == null) {
            throwAssertionError(msg);
        }
    }

    private static void throwAssertionError(String msg) {
        throwAssertionError(msg, null);
    }

    private static void throwAssertionError(String msg, Throwable t) {
        if (sShouldThrowError) {
            if (t != null) {
                Log.e("throwAssertionError", msg);
                throw new AssertionError(t);
            } else {
                throw new AssertionError(msg);
            }
        } else {
            CameraLogger.uploadError(msg, Log.getStackTraceString(t));
        }
    }

    public static void beNullIf(Object obj, boolean ifCondition) {
        beNullIf(obj, ifCondition, "beNullIf assert fail");
    }

    public static void beNullIf(Object obj, boolean ifCondition, String msg) {
        if (ifCondition && obj != null) {
            throwAssertionError(msg);
        }
    }

    public static void notEmpty(CharSequence str) {
        notEmptyIf(str, true, "notEmpty assert fail");
    }

    public static void notEmptyIf(CharSequence str, boolean ifCondition) {
        notEmptyIf(str, ifCondition, "notEmptyIf assert fail");
    }

    public static void notEmptyIf(CharSequence str, boolean ifCondition, String msg) {
        if (ifCondition && TextUtils.isEmpty(str)) {
            throwAssertionError(msg);
        }
    }

    public static void notEmpty(Collection c) {
        notEmptyIf(c, true, "notEmpty assert fail");
    }

    public static void notEmptyIf(Collection c, boolean ifCondition, String msg) {
        if (ifCondition && (c == null || c.isEmpty())) {
            throwAssertionError(msg);
        }
    }

    public static void beTrue(boolean b) {
        beTrueIf(b, true);
    }

    public static void beTrue(boolean b, String msg) {
        beTrueIf(b, true, msg);
    }

    /**
     * 在ifCondition为true的情况下，断言b为true
     */
    public static void beTrueIf(boolean b, boolean ifCondition) {
        beTrueIf(b, ifCondition, "beTrueIf assert fail");
    }

    public static void beTrueIf(boolean b, boolean ifCondition, String msg) {
        if (ifCondition && !b) {
            throwAssertionError(msg);
        }
    }

    public static void beFalse(boolean b, String msg) {
        beFalseIf(b, true, msg);
    }

    public static void beFalse(boolean b) {
        beFalseIf(b, true);
    }

    public static void beFalseIf(boolean b, boolean ifCondition) {
        beFalseIf(b, ifCondition, "beFalseIf assert fail");
    }

    public static void beFalseIf(boolean b, boolean ifCondition, String msg) {
        if (ifCondition && b) {
            throwAssertionError(msg);
        }
    }

    public static void fail() {
        fail("assert fail");
    }

    public static void fail(String msg) {
        throwAssertionError(msg);
    }

    public static void fail(String msg, Throwable t) {
        throwAssertionError(msg, t);
    }

    public static void beEqual(int origin, int expect) {
        if (origin != expect) {
            throwAssertionError("" + origin + " not equal to " + expect);
        }
    }
}
