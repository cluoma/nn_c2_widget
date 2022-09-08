<svelte:options tag={"mnist-convolution-checker-widget"} />

<script>
	import { matrix, multiply, round, add, dotDivide, exp, map } from "mathjs";
	import { fabric } from "fabric-pure-browser";
	import { onMount } from "svelte";
	import * as tf from "@tensorflow/tfjs";

	/* variables to hold canvases */
	let canvas_id; // id of canvas
	let scaled_canvas_id;
	let canv; // the fabric canvas
	let scaled_canv;

	/* load pre-trained model from webserver */
	var model;
	(async () => {
		model = await tf.loadLayersModel(
			"https://www.cluoma.com/extras/mnist_widget_model/model.json"
		);
	})();

	
	let prediction = null;  // the predicted number
	let pred_probs = {
		number: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
		prediction: [],  // the output activation for each digit
	};
	function clear_prediction() {
		prediction = null;
		pred_probs.prediction = [];
	}

	/* predict the number
	 *
	 * num_array: the array of grayscale pixel values from the canvas
	 *
	 * return: the most-likely number
	 */
	function predict_number(num_array) {
		let pixels = tf.reshape(tf.tensor(num_array), [1, 28, 28, 1]);
		let mod_pred = model.predict(pixels).dataSync();

		let max_num = 0;
		let max_val = 0;
		for (let i = 0; i < 10; i += 1) {
			// saves prediction strength to array for display
			pred_probs.prediction[i] = mod_pred[i];
			if (mod_pred[i] > max_val) {
				max_num = i;
				max_val = mod_pred[i];
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
	}

	/* prepare canvases */
	onMount(() => {
		canv = new fabric.Canvas(canvas_id);
		canv.isDrawingMode = true;
		canv.freeDrawingBrush.width = 26;
		canv.freeDrawingBrush.color = "#000000";
		canv.backgroundColor = "#ffffff";
		canv.renderAll();

		canv.on('mouse:down', () => {clear_prediction();});

		scaled_canv = new fabric.Canvas(scaled_canvas_id);
	});
</script>

<main>
	<div class="float-container">
		<h3>Digit Classification Tool - Convolutional Network</h3>
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
			results. A simple convolutional network model was trained using
			MNIST data.
		</p>
	</div>
</main>

<style>
	main {
		font-family: Roboto, -apple-system, BlinkMacSystemFont, Segoe UI, Oxygen,
			Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue, sans-serif;
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
