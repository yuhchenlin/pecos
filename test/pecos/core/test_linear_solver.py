#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
#  with the License. A copy of the License is located at
#
#  http://aws.amazon.com/apache2.0/
#
#  or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
#  OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.
import pytest  # noqa: F401; pylint: disable=unused-variable
from pytest import approx


def test_consistency_of_warm_start(tmpdir):
    from pecos.utils import smat_util
    from pecos.xmc import MLProblem, MLModel
    from pecos.core import (
        ScipyCscF32,
        clib,
    )

    train_sX_file = "test/tst-data/xmc/xlinear/X.npz"
    train_Y_file = "test/tst-data/xmc/xlinear/Y.npz"

    X = smat_util.load_matrix(train_sX_file)
    Y = smat_util.load_matrix(train_Y_file)

    prob = MLProblem(X, Y)
    train_params = MLModel.TrainParams(solver_type="L2R_L2LOSS_SVC_PRIMAL", threshold=0.0)

    # train W0:
    W0 = clib.xlinear_single_layer_train(
        prob.pX,
        prob.pY,
        prob.pC,
        prob.pM,
        prob.pR,
        **train_params.to_dict(),
    )
    pW0 = ScipyCscF32.init_from(W0)

    # warm start W (to compare without max_iter=0)
    train_params = MLModel.TrainParams(solver_type="L2R_L2LOSS_SVC_PRIMAL", threshold=0.0)
    W = clib.xlinear_single_layer_train(
        prob.pX,
        prob.pY,
        prob.pC,
        prob.pM,
        prob.pR,
        pW0,
        **train_params.to_dict(),
    )
    pW = ScipyCscF32.init_from(W)
    assert pW.buf.todense() == approx(pW0.buf.todense(), abs=1e-6)

    # warm start W1: with max_iter=0, check if it stops immediately taking warm start
    train_params = MLModel.TrainParams(
        solver_type="L2R_L2LOSS_SVC_PRIMAL", max_iter=0, threshold=0.0
    )
    W1 = clib.xlinear_single_layer_train(
        prob.pX,
        prob.pY,
        prob.pC,
        prob.pM,
        prob.pR,
        pW,
        **train_params.to_dict(),
    )
    pW1 = ScipyCscF32.init_from(W1)
    assert pW1.buf.todense() == approx(pW.buf.todense(), abs=1e-6)

    # warm start W2: start from W0 (to compare without max_iter=0)
    train_params = MLModel.TrainParams(solver_type="L2R_L2LOSS_SVC_PRIMAL", threshold=0.0)
    W2 = clib.xlinear_single_layer_train(
        prob.pX,
        prob.pY,
        prob.pC,
        prob.pM,
        prob.pR,
        pW0,
        **train_params.to_dict(),
    )
    pW2 = ScipyCscF32.init_from(W2)
    assert pW2.buf.todense() == approx(pW0.buf.todense(), abs=1e-6)
    assert pW2.buf.todense() == approx(pW.buf.todense(), abs=1e-6)
    assert pW2.buf.todense() == approx(pW1.buf.todense(), abs=1e-6)
