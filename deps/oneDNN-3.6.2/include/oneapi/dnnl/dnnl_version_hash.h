/*******************************************************************************
 * Copyright 2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#ifndef ONEAPI_DNNL_DNNL_VERSION_HASH_H
#define ONEAPI_DNNL_DNNL_VERSION_HASH_H

// clang-format off

/// Note: this macro and header file were moved to a separate instance to avoid
/// incremental build issues as moving from commit to commit would trigger a
/// complete library rebuild. Including a generated header file in a single
/// translation unit makes this problem go away.
/// Git commit hash
#define DNNL_VERSION_HASH  "c5291ae7b85780985b9622b231b8c643b88e6a87"

// clang-format on

#endif
