<svelte:options tag={"mnist-checker-widget"} />

<script>
	import { matrix, multiply, round, add, dotDivide, exp, map } from "mathjs";
	import { fabric } from "fabric-pure-browser";
	import { onMount } from "svelte";

	/* variables to hold canvases */
	let canvas_id; // id of canvas
	let scaled_canvas_id;
	let canv; // the fabric canvas
	let scaled_canv;

	let prediction = null;  // the predicted number
	let pred_probs = {
		number: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
		prediction: [],  // the output activation for each digit
	};

	/* sigmoid function needed for feedforward */
	function sigmoid(z) {
		var bottom = add(1, map(multiply(-1, z), exp));
		return dotDivide(1, bottom);
	}

	/* import model weights and store in matrices */
	import weights from "./weights_as_json.json";
	let biases0 = matrix(weights.biases0);
	let biases1 = matrix(weights.biases1);
	let biases2 = matrix(weights.biases2);
	let weights0 = matrix(weights.weights0);
	weights0.reshape([100, 784]);
	let weights1 = matrix(weights.weights1);
	weights1.reshape([30, 100]);
	let weights2 = matrix(weights.weights2);
	weights2.reshape([10, 30]);

	/* commented out test prediction */
	// import test_num from './test_json.json';
	// let test_number = matrix(test_num.test_number);
	// test_number = dotDivide(test_number, 255);

	// let a1 = multiply(weights0, test_number);
	// let a2 = sigmoid(add(a1, biases0));

	// let a3 = multiply(weights1, a2);
	// let a4 = sigmoid(add(a3, biases1));

	// let a5 = multiply(weights2, a4);
	// let a6 = sigmoid(add(a5, biases2));

	// let max_num = 0;
	// let max_val = 0;
	// for (let i = 0; i < 10; i += 1) {
	// 	if (a6.get([i]) > max_val) {
	// 		max_num = i;
	// 		max_val = a6.get([i]);
	// 	}
	// }


	/* predict the number by running the network
	 *
	 * num_array: the array of grayscale pixel values from the canvas
	 *
	 * return: the most-likely number
	 */
	function predict_number(num_array) {
		let test_number = matrix(num_array);

		let a1 = multiply(weights0, test_number);
		let a2 = sigmoid(add(a1, biases0));

		let a3 = multiply(weights1, a2);
		let a4 = sigmoid(add(a3, biases1));

		let a5 = multiply(weights2, a4);
		let a6 = sigmoid(add(a5, biases2));

		let max_num = 0;
		let max_val = 0;
		for (let i = 0; i < 10; i += 1) {
			//console.log(i + " " + a6.get([i]));
			pred_probs.prediction[i] = a6.get([i]);
			if (a6.get([i]) > max_val) {
				max_num = i;
				max_val = a6.get([i]);
			}
		}
		prediction = max_num;
		return max_num;
	}

	/* blanks out the main canvas */
	function clearCanvas() {
		canv.clear();
		canv.backgroundColor = "#ffffff";
		canv.renderAll();
	}

	/* callback when the 'predict' button is pressed
	 *
	 * scales the main canvas to 28x28 and get the grayscale value
	 * of pixels
	 *
	 * calls 'predict' on the grayscale pixels
	 */
	function predict() {
		canv.freeDrawingBrush._finalizeAndAddPath();

		let gfg = canvas_id.getContext("2d");
		let ctxScaled = scaled_canvas_id.getContext("2d");
		ctxScaled.save();
		ctxScaled.clearRect(
			0,
			0,
			ctxScaled.canvas.height,
			ctxScaled.canvas.width
		);
		ctxScaled.scale(28.0 / gfg.canvas.width, 28.0 / gfg.canvas.height);
		ctxScaled.drawImage(canvas_id, 0, 0);
		const { data } = ctxScaled.getImageData(0, 0, 28, 28);
		ctxScaled.restore();

		let pixels = data;
		let grayscale = Array(28 * 28);
		for (var i = 0; i < pixels.length; i += 4) {
			// let lightness = parseInt(pixels[i]*.299 + pixels[i + 1]*.587 + pixels[i + 2]*.114);
			// let lightness = parseInt(3*pixels[i] + 4*pixels[i + 1] + pixels[i + 2] >>> 3);
			let lightness =
				0.2126 * pixels[i] +
				0.715 * pixels[i + 1] +
				0.0722 * pixels[i + 2];
			grayscale[i / 4] = (255 - lightness) / 255.0;
		}

		predict_number(grayscale);

		/* Print the array on console */
		// filtered = matrix(grayscale);
		// console.log(grayscale);
		// console.log(predict_number(grayscale));
	}

	/* prepare canvases */
	onMount(() => {
		canv = new fabric.Canvas(canvas_id);
		canv.isDrawingMode = true;
		canv.freeDrawingBrush.width = 26;
		canv.freeDrawingBrush.color = "#000000";
		canv.backgroundColor = "#ffffff";
		canv.renderAll();
		scaled_canv = new fabric.Canvas(scaled_canvas_id);
	});
</script>

<main>
	<div class="float-container">
		<h3>Digit Classification Tool - Fully Connected Network</h3>
		<div class="float-child">
			<canvas bind:this={canvas_id} width="300" height="300" />
			<br />
			<button on:click={predict}>Predict</button>
			<button on:click={clearCanvas}>Clear</button>
			<canvas
				bind:this={scaled_canvas_id}
				style="display:none"
				width="28"
				height="28"
			/>
		</div>
		<div class="float-child" style="width: 180px;padding-top:0px;">
			<h3 style="padding-top:0px;margin-top:0px;">
				Prediction: <strong class="predictionnum"
					>{#if prediction !== null}{prediction}{/if}</strong
				>
			</h3>
			{#if prediction !== null}
				<table id="responses">
					<tr><th>number</th><th>prediction strength</th></tr>
					{#each pred_probs.number as number}
						<tr>
							<td>{number.toString()}</td>
							<td>{pred_probs.prediction[number].toFixed(3)}</td>
						</tr>
					{/each}
				</table>
			{/if}
		</div>
	</div>
	<div style="clear: both">
		<p>
			Draw a digit between 0-9 in the box and click predict to view
			results. A simple multilayer perceptron model was trained using
			MNIST data. Accuracy on test data was 98.2% but it performs rather
			poorly in this tool.
		</p>
	</div>
	<!-- <p>something</p>

	<p>{filtered}</p>
	<p>{test_number}</p>

	<p>{a1}</p>
	<p>{a2}</p>

	<p>{a3}</p>
	<p>{a4}</p>

	<p>{a5}</p>
	<p>{a6}</p>

	<p>{max_num}</p> -->
</main>

<style>
	main {
		font-family: Roboto, -apple-system, BlinkMacSystemFont, Segoe UI, Oxygen,
			Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue, sans-serif;
		/* position: relative; */
		max-width: 100%;
		background-color: white;
		background-color: rgb(43, 43, 43);
		padding: 1em;
		margin: 0 auto;
		box-sizing: border-box;
		color: rgb(169, 183, 198);
		margin-top: 1em;
		margin-bottom: 1em;
		border-radius: 5px;
	}
	canvas {
		border: 1px solid black;
	}

	.float-child {
		width: 320px;
		float: left;
		padding: 10px;
	}
	.predictionnum {
		font-size: 2em;
	}
</style>
