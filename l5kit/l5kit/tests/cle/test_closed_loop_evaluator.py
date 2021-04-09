import unittest
from unittest.mock import Mock

from l5kit.cle import closed_loop_evaluator as ceval
from l5kit.cle import validators


class TestEvaluationPlan(unittest.TestCase):
    def test_attributes(self) -> None:
        test_metric = Mock(metric_name="test_metric")
        test_validator = Mock(requires_metric=["test_metric"],
                              validator_name="test_validator")
        ep = ceval.EvaluationPlan([test_metric], [test_validator])
        self.assertDictEqual(ep.metrics_dict(),
                             {"test_metric": test_metric})
        self.assertDictEqual(ep.validators_dict(),
                             {"test_validator": test_validator})

    def test_inconsistency(self) -> None:
        metric_list = [Mock(metric_name="metric_a")]
        validator_list = [Mock(requires_metric=["metric_b"],
                               validator_name="test_validator")]
        with self.assertRaisesRegex(RuntimeError, "required by validator"):
            _ = ceval.EvaluationPlan(metric_list, validator_list)

    def test_evaluate(self) -> None:
        test_metric = Mock(metric_name="test_metric")
        test_validator = Mock(requires_metric=["test_metric"],
                              validator_name="test_validator")
        ep = ceval.EvaluationPlan([test_metric], [test_validator])
        ret = ep.evaluate(Mock())
        test_metric.compute.assert_called_once()
        test_validator.validate.assert_not_called()
        self.assertIn(test_metric.metric_name, ret)

    def test_validate(self) -> None:
        test_metric = Mock(metric_name="test_metric")
        test_validator = Mock(requires_metric=["test_metric"],
                              validator_name="test_validator")
        ep = ceval.EvaluationPlan([test_metric], [test_validator])
        ret = ep.validate({
            "test_metric": Mock(),
        }, Mock())
        test_validator.validate.assert_called_once()
        test_metric.compute.assert_not_called()
        self.assertIn(test_validator.validator_name, ret)

    def test_process_interventions(self) -> None:
        metric_list = [Mock(metric_name="test_metric")]
        validator_list = [
            Mock(requires_metric=["test_metric"], validator_name="test_validator_a"),
            Mock(requires_metric=["test_metric"], validator_name="test_validator_b"),
        ]
        intervention_validators = ["test_validator_b"]
        ep = ceval.EvaluationPlan(metric_list, validator_list,
                                  intervention_validators=intervention_validators)
        validation_res_mock = {
            "test_validator_a": validators.ValidatorOutput(False, [1]),
            "test_validator_b": validators.ValidatorOutput(False, [1]),
        }
        res = ep.process_interventions(validation_res_mock)
        self.assertTrue(res["test_validator_a"].is_valid_scene)
        self.assertEqual(len(res["test_validator_a"].failed_frames), 0)

    def test_process_interventions_ordering(self) -> None:
        metric_list = [Mock(metric_name="test_metric")]
        validator_list = [
            Mock(requires_metric=["test_metric"], validator_name="test_validator_a"),
            Mock(requires_metric=["test_metric"], validator_name="test_validator_b"),
            Mock(requires_metric=["test_metric"], validator_name="test_validator_c"),
        ]
        intervention_validators = ["test_validator_b", "test_validator_a"]
        ep = ceval.EvaluationPlan(metric_list, validator_list,
                                  intervention_validators=intervention_validators)
        validation_res_mock = {
            "test_validator_a": validators.ValidatorOutput(False, [30, 31, 32]),
            "test_validator_b": validators.ValidatorOutput(False, [20, 21, 22]),
            "test_validator_c": validators.ValidatorOutput(False, [10, 11, 12]),
        }

        # The test_validator_b failed ealier than test_validator_a, so it
        # should take precendence
        res = ep.process_interventions(validation_res_mock)
        self.assertTrue(res["test_validator_a"].is_valid_scene)
        self.assertEqual(len(res["test_validator_a"].failed_frames), 0)

        # The test_validator_c is not an intervention validator
        self.assertTrue(res["test_validator_c"].is_valid_scene)
        self.assertEqual(len(res["test_validator_c"].failed_frames), 0)

    def test_evaluate_composite(self) -> None:
        test_metric = Mock(metric_name="test_metric")
        test_validator = Mock(requires_metric=["test_metric"],
                              validator_name="test_validator")
        composite_metric = Mock(requires_metric=["test_metric"],
                                requires_validator=["test_validator"],
                                composite_metric_name="composite_metric")
        ep = ceval.EvaluationPlan([test_metric], [test_validator], [composite_metric])
        ret = ep.evaluate_composite(Mock(), {
            "test_metric": Mock(),
        }, {
            "test_validator": Mock(),
        })
        test_validator.validate.assert_not_called()
        test_metric.compute.assert_not_called()
        composite_metric.compute.assert_called_once()
        self.assertIn(composite_metric.composite_metric_name, ret)

    # TODO(perone): uncomment when we have validators
    # def test_metric_validator_integration(self) -> None:
    #     metric_list = [metrics.CollisionRearMetric()]
    #     validator_list = [
    #         validators.RangeValidator("rear_validator",
    #                                   metrics.CollisionRearMetric,
    #                                   max_value=0)
    #     ]
    #     eval_plan = ceval.EvaluationPlan(metric_list, validator_list)
    #     self.assertEqual(len(eval_plan.metrics_dict()), 1)
    #     self.assertEqual(len(eval_plan.validators_dict()), 1)

    def test_repeated_metrics(self) -> None:
        metric_list = [Mock(metric_name="metric_a"),
                       Mock(metric_name="metric_a")]
        with self.assertRaisesRegex(RuntimeError, "You cannot have repeated metric names."):
            _ = ceval.EvaluationPlan(metric_list)

    def test_repeated_validators(self) -> None:
        metric_list = [Mock(metric_name="metric_a")]
        validator_list = [Mock(validator_name="validator_a"),
                          Mock(validator_name="validator_a")]
        with self.assertRaisesRegex(RuntimeError, "You cannot have repeated validator names."):
            _ = ceval.EvaluationPlan(metric_list, validator_list)

    def test_repeated_composite_metrics(self) -> None:
        metric_list = [Mock(metric_name="metric_a")]
        validator_list = [Mock(validator_name="validator_a")]
        composite_metric_list = [Mock(composite_metric_name="composite_metric_a"),
                                 Mock(composite_metric_name="composite_metric_a")]
        with self.assertRaisesRegex(RuntimeError, "You cannot have repeated composite metric names."):
            _ = ceval.EvaluationPlan(metric_list, validator_list, composite_metric_list)

    def test_interventions_not_found(self) -> None:
        metric_list = [Mock(metric_name="metric_a")]
        validator_list = [Mock(validator_name="validator_a")]
        composite_metric_list = [Mock(composite_metric_name="composite_metric_a")]
        intervention_validators = ["validator_not_found"]
        with self.assertRaisesRegex(RuntimeError, "not found in validators list"):
            _ = ceval.EvaluationPlan(metric_list, validator_list,
                                     composite_metric_list, intervention_validators)
