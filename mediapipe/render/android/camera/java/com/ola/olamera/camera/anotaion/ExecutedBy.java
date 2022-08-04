package com.ola.olamera.camera.anotaion;

/*
 * Copyright 2019 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


import androidx.annotation.RestrictTo;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.util.concurrent.Executor;

/**
 * Denotes that the annotated method should only be executed by the referenced Executor or Handler.
 * <p>
 * Example:
 * <pre>
 * final Executor executor = new ThreadPoolExecutor();
 *
 * public void doSomething() {
 *     executor.execute(this::doSomethingOnExecutor);
 * }
 *
 * {@literal @}ExecutedBy("executor")
 * void doSomethingOnExecutor() {
 *     // Do something while being executed by the executor
 * }</pre>
 *
 * <p>This can be used to denote that it is not safe to call this method when not executed by a
 * specific {@link Executor}, if, for instance, the Executor provides certain guarantees of which
 * thread the code will run on or guarantees of sequential (non-concurrent) execution.
 *
 * @hide
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.SOURCE)
public @interface ExecutedBy {
    String value();
}
