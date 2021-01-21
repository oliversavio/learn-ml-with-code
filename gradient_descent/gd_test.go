package main

import (
	"fmt"
	"testing"
)

func TestLossFunction(t *testing.T) {

	act := []float64{1.0, 2.0, 3.0}
	preds := []float64{1.0, 2.0, 3.0}
	if costFunction(act, preds) != 0.0 {
		t.Error("Incorrect LossFunction, 0.0 is expected")
	}
}

func TestLossFunctionWithValues(t *testing.T) {
	act := []float64{-0.4257, -0.3371, 1.8067, 3.6526, 4.0358, 5.1206, 5.1943, 6.7924, 7.0681, 7.4090, 8.8640, 10.4774, 11.4812, 11.4987, 12.0733, 15.1279, 17.0229, 16.4442, 18.7043, 19.7099}
	preds := []float64{0.4701, 0.9774, 1.4847, 1.9920, 2.4993, 3.0066, 3.5139, 4.0212, 4.5285, 5.0358, 5.5431, 6.0504, 6.5577, 7.0650, 7.5723, 8.0796, 8.5869, 9.0942, 9.6015, 10.1088}

	// Not the best test in the world
	if costFunction(act, preds) < 23 || costFunction(act, preds) >= 24 {
		fmt.Print(costFunction(act, preds))
		t.Error("Incorrect value, 23.9751 was expected")
	}
}

func TestModelFuncF(t *testing.T) {
	in := []float64{1., 2., 3.}
	p := &params{m: 1., c: 1.}

	actuals := f(in, p)

	for i, a := range actuals {
		expected := (p.m * float64(i+1.)) + p.c
		if a != float64(expected) {
			t.Error("Model function not as expected mx + c")
		}
	}

}
