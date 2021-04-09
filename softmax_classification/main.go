package main

import "fmt"
import "gonum.org/v1/gonum/mat"
import "math"
import "math/rand"

func main() {
	SimpleNN()
	fmt.Println()
	SoftmaxNN()
}

func SimpleNN() {
	nn := linear()
	fmt.Println("Activation after linear layer")
	fmt.Println(mat.Formatted(nn))
}

func SoftmaxNN() {
	nn := linear()
	nnsmax := softmax(nn)

	fmt.Println("Activation after softmax layer")
	fmt.Println(mat.Formatted(nnsmax))
}

func initRandomMatrix(row int, col int) *mat.Dense {
	size := row * col
	arr := make([]float64, size)
	for i := range arr {
		arr[i] = rand.NormFloat64()
	}

	return mat.NewDense(row, col, arr)
}

func softmax(matrix *mat.Dense) *mat.Dense {
	var sum float64
	for _, v := range matrix.RawMatrix().Data {
		sum += math.Exp(v)
	}

	res := mat.NewDense(matrix.RawMatrix().Rows, matrix.RawMatrix().Cols, nil)
	res.Apply(func(i int, j int, v float64) float64 {
		return math.Exp(v) / sum
	}, matrix)

	return res
}

func linear() *mat.Dense {
	m := initRandomMatrix(3, 3)
	x := initRandomMatrix(3, 1)
	c := initRandomMatrix(3, 1)

	ll := mat.NewDense(3, 1, nil)
	ll.Mul(m, x)
	ll.Add(ll, c)
	return ll
}
