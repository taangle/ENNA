let chartWidth;
let chartHeight;
let chartInnerWidth;
let chartInnerHeight;
const chartMargin = { top: 40, right: 40, bottom: 40, left: 40 };

let chartSvg;
let bestLineChartSvg;
let chartG;
let generationInput;
let playButton;
let hoverDiv;

// TODO get this automatically
let lastGenerationIndex = 26;
let lambda = .1;

let allLayers;
let generations;
let lastDotPositions;

let timer;

document.addEventListener('DOMContentLoaded', () => {
    chartSvg = d3.select('#chart');
    // bestLineChartSvg = d3.select('#best-line-chart');
    chartG = chartSvg.append('g')
      .attr('transform', 'translate(' + chartMargin.left + ',' + chartMargin.top + ')');
    generationInput = d3.select('#generation-input');
    playButton = d3.select('#play-button')

    chartWidth = +chartSvg.style('width').replace('px','');
    chartHeight = +chartSvg.style('height').replace('px','');
    chartInnerWidth = chartWidth - chartMargin.left - chartMargin.right;
    chartInnerHeight = chartHeight - chartMargin.top - chartMargin.bottom;

    console.log("About to load data");
    d3.csv('../histories/20210722-001843/history.csv').then(rawLayers => {
        console.log("Loaded data");
        for (layer of rawLayers) {
            layer["generation"] = +layer["generation"]
            layer["genome"] = +layer["genome"]
            layer["accuracy"] = +layer["accuracy"]
            layer["training_time"] = +layer["training_time"]
            layer["layer"] = +layer["layer"]
            layer["units"] = +layer["units"]
        }
        allLayers = rawLayers;
           
        generationInput.node().addEventListener('input', changeGeneration);
        playButton.node().addEventListener('click', clickPlayButton);

        drawChart(); 

        // drawBestLineChart();
    });
});

function timeStep() {
    if (+generationInput.node().value < lastGenerationIndex) {
        generationInput.property('value', (+(generationInput.node().value)) + 1);
        drawChart();
    }
}

function clickPlayButton() {
    if (playButton.text() === 'Pause') {
        clearInterval(timer);
        playButton.text('Play');
    }
    else {
        timer = setInterval(timeStep, 1000);
        playButton.text('Pause');
    }
}

function changeGeneration() {
    drawChart();
}

// TODO line chart is messed up, with duplicate code to boot
function drawBestLineChart() {
    let data = [];
    let generations = [...Array(lastGenerationIndex + 1).keys()]
    let bestAccuracies = [];
    for (let g = 0; g <= lastGenerationIndex; g++) {
        let thisGeneration = allLayers.filter(l => l["generation"] === g);
        let bestGenomeIndex = -1;
        let maxFitness = 0;
        let bestAccuracy = 0;
        for (let layer of thisGeneration) {
            accuracy = layer['accuracy'];
            trainingTime = layer['training_time'];
            // TODO I know this is dumb, but if this changes, change this
            // or just add fitness to history
            inverse = lambda * (1.0 / trainingTime);
            fitness = accuracy + inverse;
            if (fitness > maxFitness) {
                maxFitness = fitness;
                bestAccuracy = accuracy;
                bestGenomeIndex = (layer["genome"])
            }
        }
        bestAccuracies.push(bestAccuracy);
    }

    for (let i = 0; i <= lastGenerationIndex; i++) {
        data.push({
            generation: generations[i],
            bestAccuracy: bestAccuracies[i]
        });
    }

    let thischartInnerWidth = 800 - chartMargin.left - chartMargin.right;
    let thischartInnerHeight = 800 - chartMargin.top - chartMargin.bottom;

    let x = d3.scaleLinear()
        .domain([0, lastGenerationIndex])
        .range([0, thischartInnerWidth]);
    
    let y = d3.scaleLinear()
        .domain([0, 1])
        .range([0, thischartInnerHeight]);

    bestLineChartSvg.append("g")
        .attr("transform", `translate(${chartMargin.left}, ${chartMargin.top + thischartInnerHeight})`)
        .call(d3.axisBottom(x));
    
    bestLineChartSvg.append("g")
        .attr("transform", `translate(${chartMargin.left}, ${chartMargin.top})`)
        .call(d3.axisLeft(y));

    // let g = bestLineChartSvg.append("g")

    bestLineChartSvg.append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-width", 1.5)
        .attr("translate", `transform(${chartMargin.left}, ${chartMargin.top + thischartInnerHeight})`)
        .attr("d", d3.line()
            .x(d => x(d.generation))
            .y(d => y(d.bestAccuracy)));
}

function drawChart() {
    // TODO bad bad bad
    for (let i = 0; i < 24; i++) {
        let dots = document.getElementsByClassName(constructLayerDotHtmlClass(i));
        while (dots[0]) dots[0].remove();
    }

    let lines = document.getElementsByClassName(".weight-line");
    while (lines[0]) lines[0].remove();

    let generationIndex = +generationInput.node().value;
    console.log(generationIndex);

    let bestGenomeIndex = -1;
    let maxFitness = 0;
    // TODO is there a more efficient way to do this?
    for (let layer of allLayers.filter(l => l['generation'] === generationIndex)) {
        accuracy = layer['accuracy'];
        trainingTime = layer['training_time'];
        // TODO if this changes, change this
        // or just add fitness to history
        inverse = lambda * (1.0 / trainingTime);
        fitness = accuracy + inverse;
        if (fitness > maxFitness) {
            maxFitness = fitness;
            bestGenomeIndex = (layer["genome"])
        }
    }

    bestLayers = allLayers.filter(l => (l['generation'] === generationIndex) && (l['genome'] === bestGenomeIndex))
    bestNumberOfLayers = bestLayers.length

    let xScale = d3.scaleLinear()
        // the domain is the x *values*, so in this case the layer indeces of the genome
        .domain([0, bestNumberOfLayers])
        // the range is the *functions* of the *values of x*, so is in this case goes over the inner width of our chart
        .range([0, chartInnerWidth]);
    numberOfNodesInWidestDeepLayerAmongAllShallowLayers = Math.max(...(bestLayers.map(l => l['units'])))
    let yScale = d3.scaleLinear()
        .domain([0, numberOfNodesInWidestDeepLayerAmongAllShallowLayers])
        .range([0, chartInnerHeight]);
    
    console.log("bestLayers:");
    console.log(bestLayers)
    let previousLayer;
    let layerIndex = 0;
    for (layer of bestLayers) {
        let unitCount = layer['units'];

        let indeces = Array(unitCount).keys();
        let dotsAtThisLayer = chartG.selectAll(constructLayerDotHtmlClass(layerIndex))
            .data([...indeces], i => i)
            .join(
                enter => enterDots(enter, layer, layerIndex, previousLayer, xScale, yScale),
                update => updateDots(update, layer, layerIndex, previousLayer, xScale, yScale),
                exit => exitDots(exit)
            );
        
        previousLayer = layer;
        layerIndex++;
    }

    chartG.select('text').remove();
    chartG.select('text').remove();
    chartG.select('text').remove();

    chartG.append('text')
        .attr('class', 'info')
        .attr('x', chartWidth / 2)
        .attr('y', chartInnerHeight + 15)
        .text(`Fitness: ${maxFitness}`);
    
    chartG.append('text')
        .attr('class', 'info')
        .attr('x', chartWidth / 2)
        .attr('y', chartInnerHeight + 35)
        .text(`Accuracy: ${bestLayers[0]["accuracy"]}`);
    
    if (generationIndex >= 24) {
        chartG.append('text')
            .attr('class', 'info')
            .attr('x', chartWidth / 4)
            .attr('y', chartInnerHeight + 15)
            .text("Elitism enabled");
    }
    // TODO idea: scale size of dots to size of weights leading into them at their deep layer
}

function constructLayerDotHtmlClass(layerIndex) {
    return `.layer-${layerIndex}-dot`
}

function enterDots(enter, layer, layerIndex, previousLayer, xScale, yScale) {
    if (layerIndex === 0) {
        enter.append('g')
            .attr('class', constructLayerDotHtmlClass(layerIndex))
            .attr('transform', (d, i) => {
                return `translate(${xScale(layerIndex)}, ${yScale(i)})`;
            })
            .style('opacity', 0)
        .call(g => g
            .transition()
            .duration(250)
            .style('opacity', 1)
        )
        .call(g => g.append('circle')
            .attr('r', 4)
            // TODO color??
            .style('fill', 'black')
            .style('stroke', 'black')
        );
    }
    else {
        enter.append('g')
            .attr('class', constructLayerDotHtmlClass(layerIndex))
            .attr('transform', (d, i) => {
                return `translate(${xScale(layerIndex)}, ${yScale(i)})`;
            })
            .style('opacity', 0)
        .call(g => g
            .transition()
            .duration(500)
            .style('opacity', 1)
        )
        .call(g => g.append('circle')
            .attr('r', 4)
            // TODO color??
            .style('fill', 'black')
            .style('stroke', 'black')
        )
        .call(g => {
            for (let k = 0; k < previousLayer["units"]; k++) {
                let x2 = xScale(layerIndex - 1) - xScale(layerIndex);
                g.append('line')
                    .style("stroke", "black")
                    .style("stroke-width", 1)
                    .attr("x1", 0)
                    .attr("y1", 0)
                    .attr("x2", x2)
                    .attr("y2", (d, i) => yScale(k) - yScale(i));
            }
        });
    }
}

function updateDots(update, layer, layerIndex, previousLayer, xScale, yScale) {
    update
      .call(g => g
        .style('opacity', 1)
        .transition()
        .delay(250)
        .duration(150)
        .attr('transform', (d, i) => `${xScale(layerIndex)}, ${yScale(i)})`)
      );
}

function exitDots(exit) {
    exit
      .call(g => g
        .transition()
        .duration(250)
        .style('opacity', 0)
        .remove()
      );
}
