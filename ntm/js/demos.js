const memory = [];
const demo4KeyVector = [0., 0., 0., 0.2, 0.,0.];
const demo2EraseVector = [1., 1., 1., 1., 1., 1.,];
let demo2WeightVector = [1., 0., 0., 0., 0.,];

let demo3WeightVector = [1., 0., 0., 0., 0.,];
let demo3EraseVector = [1., 1., 1., 1., 1., 1.];
let demo3WriteVector = [0., 1., 0., 0., 0., 0.,];

let demo5WeightVector = [1., 0., 0., 0., 0.,];
let demo5Idx = 1;

function updateDemo4(keyVector, memory) {
    for (let i = 0; i < keyVector.length; i++) {
        $(`#4-kv-${i+1}`).attr('fill', getColor(keyVector[i]));
    };

    const weightingVector = [];
    for (let mem = 0; mem < 5; mem++) {
        const memSlot = memory[mem];
        const dot = dotProd(keyVector, memSlot) / (magnitude(keyVector) * magnitude(memSlot));

        weightingVector.push(dot);

        $(`#4-unw-${mem+1}`).attr('fill', getColor(dot));
    }

    const normalizedWeightingVector = softmax(weightingVector);
    const strengthenedWeightingVector = softmax(weightingVector, strengthen=10);
    for (let mem = 0; mem < 6; mem++) {
        $(`#4-nw-${mem+1}`).attr('fill', getColor(normalizedWeightingVector[mem]));
        $(`#4-sw-${mem+1}`).attr('fill', getColor(strengthenedWeightingVector[mem]));
    }
}

function updateDemo1(weightVector, memory) {
    for (let i = 0; i < weightVector.length; i++) {
        $(`#3-wv-${i+1}`).attr('fill', getColor(weightVector[i]));
    };

    for (let c = 0; c < 6; c++) {
        let sum = 0;
        for (let mem = 0; mem < 5; mem++) {
            const prod = memory[mem][c] * weightVector[mem];
            $(`#3-rm-${mem+1}${c+1}`).attr('fill', getColor(prod));

            sum += prod;
        }

        $(`#3-rv-${c+1}`).attr('fill', getColor(sum));
    }
}

function updateDemo2(weightVector, eraseVector, memory) {
    for (let i = 0; i < weightVector.length; i++) {
        $(`#2-wv-${i+1}`).attr('fill', getColor(weightVector[i]));
    };

    for (let i = 0; i < eraseVector.length; i++) {
        $(`#2-ev-${i+1}`).attr('fill', getColor(eraseVector[i]));
    };

    for (let c = 0; c < 6; c++) {
        let sum = 0;
        for (let mem = 0; mem < 5; mem++) {
            const prod = weightVector[mem] * eraseVector[c];
            $(`#2-em-${mem+1}${c+1}`).attr('fill', getColor(prod));
            $(`#2-iem-${mem+1}${c+1}`).attr('fill', getColor(1 - prod));
            $(`#2-emm-${mem+1}${c+1}`).attr('fill', getColor((1 - prod) * memory[mem][c]));

        }
    }
}

function updateDemo3(weightVector, eraseVector, writeVector, memory) {
    for (let i = 0; i < weightVector.length; i++) {
        $(`#1-wv-${i+1}`).attr('fill', getColor(weightVector[i]));
        $(`#1-wv-${i+1}-2`).attr('fill', getColor(weightVector[i]));

    };

    for (let i = 0; i < eraseVector.length; i++) {
        $(`#1-ev-${i+1}`).attr('fill', getColor(eraseVector[i]));
    };

    for (let i = 0; i < writeVector.length; i++) {
        $(`#1-wrv-${i+1}`).attr('fill', getColor(writeVector[i]));
    };

    for (let c = 0; c < 6; c++) {
        let sum = 0;
        for (let mem = 0; mem < 5; mem++) {
            const prod = weightVector[mem] * eraseVector[c];
            $(`#1-em-${mem+1}${c+1}`).attr('fill', getColor(prod));
            $(`#1-iem-${mem+1}${c+1}`).attr('fill', getColor(1 - prod));
            $(`#1-emm-${mem+1}${c+1}`).attr('fill', getColor((1 - prod) * memory[mem][c]));

            $(`#1-wm-${mem+1}${c+1}`).attr('fill', getColor(weightVector[mem] * writeVector[c]));
            $(`#1-wmm-${mem+1}${c+1}`).attr('fill', getColor(weightVector[mem] * writeVector[c] + (1 - prod) * memory[mem][c]));

        }
    }
}

function updateDemo5(weightVector) {
    for (let i = 0; i < weightVector.length; i++) {
        $(`#5-wv-${i+1}`).attr('fill', getColor(weightVector[i]));
        $(`#5-wh-${i+1}${demo5Idx+1}`).attr('fill', getColor(weightVector[i]));

    };

    demo5Idx++;
}

$(document).ready(() => {
    setTimeout(() => {
        $('#loading').css('display', 'none');
    }, 1500);
    
    setTimeout(() => {
    
        for (let r = 0; r < 5; r++) {
            const vec = [];
            for (let c = 0; c < 6; c++) {
                const cell = Math.random();

                $(`#4-mm-${r+1}${c+1}`).attr('fill', getColor(cell));
                $(`#3-mm-${r+1}${c+1}`).attr('fill', getColor(cell));
                $(`#2-mm-${r+1}${c+1}`).attr('fill', getColor(cell));
                $(`#1-mm-${r+1}${c+1}`).attr('fill', getColor(cell));

                if (c == 0) {
                    $(`#5-wh-${r+1}${c+1}`).attr('fill', getColor(demo5WeightVector[r]));
                } else {
                    $(`#5-wh-${r+1}${c+1}`).attr('fill', '#fff');
                }
                $(`#5-wh-${r+1}${c+7}`).attr('fill', '#fff');

                vec.push(cell);
            }
            memory.push(vec);
        }

        $("#demo-4").mousedown(function(e) {
            console.log(e);
            const demoWidth = 1900;
            const demoHeight = 1200;

            const offset = getOffset(e, this);
            const scale = demoWidth / $(this).width();

            console.log(scale, offset[0] * scale)
            if (between(offset[0] * scale, 000, 600) && 
                between(offset[1] * scale, 200, 300)) {
                const idx = Math.round(((offset[0] * scale - 50) / 100));

                demo4KeyVector[idx] += 0.2;
                if (demo4KeyVector[idx] > 1) {
                    demo4KeyVector[idx] = 0;
                }

                updateDemo4(demo4KeyVector, memory);
            }          
        })

        $("#demo-1").mousemove(function(e) {
            const demoWidth = 3000;
            const demoHeight = 600;

            const offset = getOffset(e, this);
            const scale = demoWidth / $(this).width();

            if (between(offset[0] * scale, 100, 200) && 
                between(offset[1] * scale, 000, 500)) {
                const idx = ((offset[1] * scale - 50) / 100);

                const attention = [];
                for (let i = 0; i < 5; i++) {
                    attention.push(1 - Math.abs(i - idx)/5);
                }               

                updateDemo1(softmax(attention, strengthen=15), memory);
            }         
        });

        $("#demo-2").mousemove(function(e) {
            const demoWidth = 3700;
            const demoHeight = 1200;

            const offset = getOffset(e, this);
            const scale = demoWidth / $(this).width();

            if (between(offset[0] * scale, 100, 200) && 
                between(offset[1] * scale, 000, 500)) {
                const idx = ((offset[1] * scale - 50) / 100);

                for (let i = 0; i < 5; i++) {
                    demo2WeightVector[i] = 1 - Math.abs(i - idx)/5;
                }               

                demo2WeightVector = softmax(demo2WeightVector, strengthen=15);

                updateDemo2(demo2WeightVector, demo2EraseVector, memory);
            }         
        });

        $("#demo-2").mousedown(function(e) {
            const demoWidth = 3700;
            const demoHeight = 1200;

            const offset = getOffset(e, this);
            const scale = demoWidth / $(this).width();

            if (between(offset[0] * scale, 400, 1000) && 
                between(offset[1] * scale, 200, 300)) {
                const idx = Math.round((offset[0] * scale - 400 - 50) / 100);

                demo2EraseVector[idx] += 0.2;
                if (demo2EraseVector[idx] > 1) {
                    demo2EraseVector[idx] = 0;
                }

                updateDemo2(demo2WeightVector, demo2EraseVector, memory);
            }         
        });


        $("#demo-3").mousemove(function(e) {
            const demoWidth = 4600;
            const demoHeight = 1950;

            const offset = getOffset(e, this);
            const scale = demoWidth / $(this).width();

            if (between(offset[0] * scale, 100, 200) && 
                (between(offset[1] * scale, 000, 500) || between(offset[1] * scale, 1450, 1950))) {

                let off = 0;
                if (offset[1] * scale > 500) {
                    off = 1450;
                }

                const idx = ((offset[1] * scale - off -50) / 100);

                for (let i = 0; i < 5; i++) {
                    demo3WeightVector[i] = 1 - Math.abs(i - idx)/5;
                }               

                demo3WeightVector = softmax(demo3WeightVector, strengthen=15);

                updateDemo3(demo3WeightVector, demo3EraseVector, demo3WriteVector, memory);
            }         
        });

        $("#demo-3").mousedown(function(e) {
            const demoWidth = 4600;
            const demoHeight = 1950;

            const offset = getOffset(e, this);
            const scale = demoWidth / $(this).width();

            if (between(offset[0] * scale, 400, 1000) && 
                between(offset[1] * scale, 200, 300)) {
                const idx = Math.round((offset[0] * scale - 400 - 50) / 100);

                demo3EraseVector[idx] += 0.2;
                if (demo3EraseVector[idx] > 1) {
                    demo3EraseVector[idx] = 0;
                }

                updateDemo3(demo3WeightVector, demo3EraseVector, demo3WriteVector, memory);
            }
            
            if (between(offset[0] * scale, 400, 1000) &&
                between(offset[1] * scale, 1650, 1750)) {
                const idx = Math.round((offset[0] * scale - 400 - 50) / 100);

                demo3WriteVector[idx] += 0.2;
                if (demo3WriteVector[idx] > 1) {
                    demo3WriteVector[idx] = 0;
                }

                updateDemo3(demo3WeightVector, demo3EraseVector, demo3WriteVector, memory); 
            }
        });

        $("#demo-5").mousedown(function(e) {
            const demoWidth = 2800;
            const demoHeight = 605;

            const offset = getOffset(e, this);
            const scale = demoWidth / $(this).width();

            if (between(offset[0] * scale, 0, 600) && 
                between(offset[1] * scale, 200, 300)) {
                const idx = Math.round((offset[0] * scale - 50) / 100);

                if (idx == 0) {
                    demo5WeightVector.push(demo5WeightVector[0]);
                    demo5WeightVector = demo5WeightVector.slice(1, demo5WeightVector.length);
                } else if (idx == 4) {
                    const temp = demo5WeightVector[demo5WeightVector.length - 1];
                    demo5WeightVector = demo5WeightVector.slice(0, demo5WeightVector.length - 1);
                    demo5WeightVector.unshift(temp);
                } 

                if (idx == 0 || idx == 2 || idx == 4) {
                    updateDemo5(demo5WeightVector);
                }
            }
        });


        updateDemo1([1., 0., 0., 0., 0.], memory);
        updateDemo2(demo2WeightVector, demo2EraseVector, memory);
        updateDemo3(demo3WeightVector, demo3EraseVector, demo3WriteVector, memory);
        updateDemo4(demo4KeyVector, memory);
        updateDemo5(demo5WeightVector);
    }, 1000);
})


let sequence = [];
let demo6Progress = 0;
let demo6MemorySlots = 0;
let demo6SequenceLength = 15;

let demo6_read_vector = generateMatrix(1, 10);
let demo6_external_input = generateMatrix(1, 7);

let demo6_prev_weights = [generateMatrix(1, 20), generateMatrix(1, 20)];
demo6_prev_weights[0][0][0] = 1.;
demo6_prev_weights[1][0][0] = 1.;

let demo6_ntm_memory = generateMatrix(20, 10);

function initializeCopyTask(memory_slots, sequence_length) {
    demo6MemorySlots = memory_slots;
    demo6SequenceLength = sequence_length;

    demo6_read_vector = generateMatrix(1, 10);
    demo6_external_input = generateMatrix(1, 7);

    demo6_prev_weights = [generateMatrix(1, memory_slots), generateMatrix(1, memory_slots)];
    demo6_prev_weights[0][0][0] = 1.;
    demo6_prev_weights[1][0][0] = 1.;

    demo6_ntm_memory = generateMatrix(memory_slots, 10);

    demo6Progress = 0;

    sequence = [];

    $("#copy-task-input").empty();
    $('#copy-task-output').empty();

    $("#copy-task-memory").empty();
    $('#copy-task-reads').empty();
    $('#copy-task-writes').empty();

    let height = 200 / memory_slots;

    for (let i = 0; i < 10; i++) {
        for (let j = 0; j < memory_slots; j++) {
            $('#copy-task-memory').append(
                $(`<div class='cell-small' id='cp-mem-${i}-${j}' style='height:${height}px'>`)
            )
        }
    }

    let width = 10 * 32/(demo6SequenceLength * 2 + 2);

    for (let i = 0; i < demo6SequenceLength * 2 + 2; i++) {
        for (let j = 0; j < memory_slots; j++) {
        $('#copy-task-reads').append(
                $(`<div class='cell-small ' id='cp-reads-${i}-${j}'  style='height:${height}px; width: ${width}px'>`)
            );
            
            $('#copy-task-writes').append(
                $(`<div class='cell-small ' id='cp-writes-${i}-${j}'  style='height:${height}px; width: ${width}px'>`)
            ) 
        }
    }
}

initializeCopyTask(20, 15);

$('#copy-task-clear').mousedown(() => {
    initializeCopyTask(40, 30);
});

$('#copy-task-mem-slots-inp').focusout(function () {
    let mem_slots = $(this).val() * 1.;
    if (mem_slots > 40) {
        mem_slots = 40;
        $(this).val(40);
    } else if (mem_slots < 1) {
        mem_slots = 1;
        $(this).val(1);
    }
    initializeCopyTask(mem_slots, demo6SequenceLength);
})
$('#copy-task-seq-length-inp').focusout(function () {
    let seq_length = $(this).val() * 1.;
    if (seq_length > 30) {
        seq_length = 30;
        $(this).val(30);
    } else if (seq_length < 1) {
        seq_length = 1;
        $(this).val(1);
    }
    initializeCopyTask(demo6MemorySlots, seq_length);
})
$('#ar-task-mem-slots-inp').focusout(function () {
    let mem_slots = $(this).val() * 1.;
    if (mem_slots > 40) {
        mem_slots = 40;
        $(this).val(40);
    } else if (mem_slots < 1) {
        mem_slots = 1;
        $(this).val(1);
    }
    initializeARTask(mem_slots, demo7NumItems);
})
$('#ar-task-num-items-inp').focusout(function () {
    let num_items = $(this).val() * 1.;
    if (num_items > 8) {
        seq_length = 8;
        $(this).val(8);
    } else if (num_items < 2) {
        num_items = 2;
        $(this).val(num_items);
    }
    initializeARTask(demo7MemorySlots, num_items);
})

$('#copy-task-step').mousedown(() => {
    let col = [];
    if (demo6Progress == demo6SequenceLength) {
        col = [0., 0., 0., 0., 0., 0., 1.];    
        for (let j = 0; j < 6;j ++) {
            $('#copy-task-input').append(
                $(`<div class='cell'></div>`)
            ) 
        }
        $('#copy-task-input').append(
            $(`<div class='cell filled' style='background: #B20404'></div>`)
        ) 
    } else if (demo6Progress < demo6SequenceLength) { 
        for (let j = 0; j < 6; j++) {
            if (Math.random() > 0.5) {
                $('#copy-task-input').append(
                    $(`<div class='cell filled' style='background: #6B68E5'></div>`)
                )
                col.push(1.);
            } else {
                $('#copy-task-input').append(
                    $(`<div class='cell'></div>`)
                )
                col.push(0.);
            }
        }

        $('#copy-task-input').append(
            $(`<div class='cell''></div>`)
        );

        col.push(0.);
        sequence.push(col);
    } else if (demo6Progress > demo6SequenceLength) {
        col = [0., 0., 0., 0., 0., 0., 0.];
    }


    console.log(demo6Progress);

    let result = ntmForward([col], demo6_ntm_memory, demo6_read_vector, demo6_prev_weights, params);
    demo6_ntm_memory = result[0];
    demo6_read_vector = result[1];
    demo6_prev_weights = result[2];

    const other_data = result[4]
    
    //[0][3][0][0];

    const read_interpolation = other_data[0][3][0][0];
    const write_interpolation = other_data[1][3][0][0];

    for (let i = 0; i < demo6MemorySlots; i++) {
        for (let j = 0; j < 10; j++) {
            $(`#cp-mem-${j}-${i}`).css('background', getColor(demo6_ntm_memory[i][j], true));
        }

        /*
        if (demo6_prev_weights[0][0][i] > 0.5) {
            if (read_interpolation < 0.3) {     // is location based
                $(`#cp-reads-${demo6Progress}-${i}`).css('border', 'red 1px solid');
            } else {    // is content based
                $(`#cp-reads-${demo6Progress}-${i}`).css('border', 'blue 1px solid'); 
            }
        }*/

        $(`#cp-reads-${demo6Progress}-${i}`).css('background', getColor(demo6_prev_weights[0][0][i]));
        $(`#cp-writes-${demo6Progress}-${i}`).css('background', getColor(demo6_prev_weights[1][0][i]));
    }

    for (let i = 0; i < 7; i++) {
        let value = Math.min(1, Math.max(0, result[3][0][i]));

        $('#copy-task-output').append(
            $(`<div class='cell' style='background: ${getColor(value, false, [50, 124, 115])}'>`)
        );
    }

    
    demo6Progress++;

});

let demo7InputSequence = [];
let demo7Items = [];
let demo7Progress = 0;
let demo7MemorySlots = 0;
let demo7NumItems = 3;

let demo7_read_vector = generateMatrix(1, 10);
let demo7_external_input = generateMatrix(1, 7);

let demo7_prev_weights = [generateMatrix(1, 20), generateMatrix(1, 20)];
demo7_prev_weights[0][0][0] = 1.;
demo7_prev_weights[1][0][0] = 1.;

let demo7_ntm_memory = generateMatrix(20, 10);

function initializeARTask(memory_slots, num_items) {
    demo7MemorySlots = memory_slots;
    demo7NumItems = num_items;
    
    demo7InputSequence = [];
    demo7Items = [];

    // generate the input sequence
    for (let it = 0; it < demo7NumItems; it ++) {
        let item = apply(generateMatrix(3, 4), (x) => (Math.random() > 0.5) * 1.);
        demo7Items.push(item);

        demo7InputSequence.push([0., 0., 0., 0., 1., 0.]);

        for (let i = 0; i < 3; i++) {
            let col = item[i];
            demo7InputSequence.push(col.concat([0., 0.]));
        }
    }

    // pick a random item
    let idx = Math.floor(Math.random() * (demo7NumItems - 1));
    let queryItem = demo7Items[idx];
    
    demo7InputSequence.push([0., 0., 0., 0., 0., 1.]);

    for (let i = 0; i < 3; i++) {
        let col = queryItem[i];
        demo7InputSequence.push(col.concat([0., 0.,]));
    }
    demo7InputSequence.push([0., 0., 0., 0., 0., 1.]);

    demo7_read_vector = generateMatrix(1, 30);
    demo7_external_input = generateMatrix(1, 7);

    demo7_prev_weights = [];
    for (let i = 0; i < 6; i++) {
        demo7_prev_weights.push(
            generateMatrix(1, demo7MemorySlots)
        )

        demo7_prev_weights[i][0][0] = 1.;
    }

    demo7_ntm_memory = generateMatrix(demo7MemorySlots, 10);

    demo7Progress = 0;

    sequence = [];

    $("#ar-task-input").empty();
    $('#ar-task-output').empty();

    $("#ar-task-memory").empty();
    $('#ar-task-reads').empty();
    $('#ar-task-writes').empty();

    let height = 200 / memory_slots;

    for (let i = 0; i < 10; i++) {
        for (let j = 0; j < memory_slots; j++) {
            $('#ar-task-memory').append(
                $(`<div class='cell-small' id='ar-mem-${i}-${j}' style='height:${height}px'>`)
            )
        }
    }

    let width = 10 * 32/(demo7NumItems * 5 + 8);

    for (let i = 0; i < demo7NumItems * 5 + 8; i++) {
        for (let j = 0; j < memory_slots; j++) {
        $('#ar-task-reads').append(
                $(`<div class='cell-small ' id='ar-reads-${i}-${j}'  style='height:${height}px; width: ${width}px'>`)
            );
            
            $('#ar-task-writes').append(
                $(`<div class='cell-small ' id='ar-writes-${i}-${j}'  style='height:${height}px; width: ${width}px'>`)
            ) 
        }
    }
}

initializeARTask(20, 4);

$('#ar-task-clear').mousedown(() => {
    initializeARTask(demo7MemorySlots, demo7NumItems);
})

$('#ar-task-step').mousedown(() => {
    let inp;
    if (demo7Progress < demo7InputSequence.length) {
        inp = demo7InputSequence[demo7Progress];
    } else {
        inp = [0., 0., 0., 0., 0., 0.];
    }

    for (let i = 0; i < 7; i++) {
        let interpolate = [0, 0, 0];

        if (inp[inp.length - 1] === 1.) {
            interpolate = [178, 4, 4];
        } else if (inp[inp.length - 2] !== 1.) {
            interpolate = [107, 104, 229];
        }

        $(`#ar-task-input`).append(
            $(`<div class='cell' style='background: ${getColor(inp[i], false, interpolate=interpolate)}'>`)
        )
    }
    let result = ntmForward([inp], demo7_ntm_memory, demo7_read_vector, demo7_prev_weights, ar_params, write_heads=3, read_heads=3);
    demo7_ntm_memory = result[0];
    demo7_read_vector = result[1];
    demo7_prev_weights = result[2];

    console.log(result[3]);

    for (let i = 0; i < demo7MemorySlots; i++) {
        for (let j = 0; j < 10; j++) {
            $(`#ar-mem-${j}-${i}`).css('background', getColor(demo7_ntm_memory[i][j], true));
        }

        $(`#ar-reads-${demo7Progress}-${i}`).css('background', getColor((demo7_prev_weights[0][0][i] + demo7_prev_weights[1][0][i] + demo7_prev_weights[2][0][i])/3.));
        $(`#ar-writes-${demo7Progress}-${i}`).css('background', getColor((demo7_prev_weights[3][0][i] + demo7_prev_weights[4][0][i] + demo7_prev_weights[5][0][i])/3.));
    }

    for (let i = 0; i < 7; i++) {
        let value = Math.min(1, Math.max(0, result[3][0][i]));
        let interpolate = [0, 0, 0];

        if (demo7Progress >= demo7InputSequence.length && demo7Progress < demo7InputSequence.length + 3) {
            interpolate = [50, 124, 115];
        }

        $('#ar-task-output').append(
            $(`<div class='cell' style='background: ${getColor(value, false, interpolate)}'>`)
        );
    }

    
    demo7Progress++;
 
})