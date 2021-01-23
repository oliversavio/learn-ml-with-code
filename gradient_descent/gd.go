package main

import (
	"fmt"
	"math"
	"math/rand"
)

const (
	lr     = 1e-3
	epochs = 15
)

type params struct {
	m float64
	c float64
}

var input_x = []float64{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.}
var actuals = []float64{-0.4257, -0.3371, 1.8067, 3.6526, 4.0358, 5.1206, 5.1943, 6.7924, 7.0681, 7.4090, 8.8640, 10.4774, 11.4812, 11.4987, 12.0733, 15.1279, 17.0229, 16.4442, 18.7043, 19.7099}

// Equation for a straight line i.e y = mx + c or y = mx + b
func f(x []float64, p *params) []float64 {
	var result []float64
	for _, val := range x {
		y := p.m*val + p.c
		result = append(result, y)
	}
	return result
}

// MSE or Mean Squared Error
func costFunction(predictions []float64) float64 {
	var diff float64
	for i := range actuals {
		d := predictions[i] - actuals[i]
		diff += math.Pow(d, 2.0)
	}
	loss := diff / float64(len(actuals))
	return loss
}

func calcGradientM(predicted []float64) float64 {
	var diff float64
	for i, x := range input_x {
		diff += (predicted[i] - actuals[i]) * x
	}
	return (diff / float64(len(input_x))) * 2
}

func calcGradientC(predicted []float64) float64 {
	var diff float64
	for i := range actuals {
		diff += (predicted[i] - actuals[i])
	}
	return (diff / float64(len(actuals))) * 2
}

func updateParams(p *params, gradientM float64, gradientC float64) {
	p.m -= lr * gradientM
	p.c -= lr * gradientC
}

func calculateLoss(x []float64, p *params) (loss float64, predicted []float64) {
	predicted = f(x, p)
	loss = costFunction(predicted)
	return
}

func calculateGradients(predicted []float64) (gradM float64, gradC float64) {
	gradM = calcGradientM(predicted)
	gradC = calcGradientC(predicted)
	return
}

func fit(p *params) {
	loss, preds := calculateLoss(input_x, p)
	fmt.Printf("Loss: %f m: %f c:%f", loss, p.m, p.c)
	fmt.Println()
	gradM, gradC := calculateGradients(preds)
	updateParams(p, gradM, gradC)
}

func train(p *params, epochs int) {
	for i := 0; i < epochs; i++ {
		fit(p)
	}
}

func main() {
	rand.Seed(7.0)
	p := &params{m: rand.Float64(), c: rand.Float64()}
	train(p, epochs)
}
