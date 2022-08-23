// const params = require('./ntm-params-800.js');

function matmul(a, b, name='') {
    if (a[0].length != b.length) {
        console.log("matmul dimensions don't match", `(${a.length}, ${a[0].length}) @ (${b.length}, ${b[0].length})`, name);
        return;
    }

    let matrix = [];
    for (let row = 0; row < a.length; row++) {
        let result_row = [];
        for (let col = 0; col < b[0].length; col++) {
            let dot = 0;
            for (let i = 0; i < b.length; i++) {
                dot += a[row][i] * b[i][col];
            }

            result_row.push(dot);
        } 

        matrix.push(result_row);
    }

    return matrix;
}

function linear(a, params, prefix) { 
    //console.log('linear with prefix ', prefix);
    let r = matmul(a, transpose(params[prefix + '.weight']), prefix);

    // r -> 1 x N
    return apply(r, (x, i, j) => x + params[prefix + '.bias'][j]);
}

function softplus(x) {
    return Math.log(1 + Math.exp(x));
}

function sigmoid(x) {
    return 1/(1 + Math.exp(-x));
}

function softmax_(x) {
    let sum = 0;
    for (let i = 0; i < x.length; i++) {
        sum += Math.exp(x[i]);
    }

    let result = [];

    for (let i = 0; i < x.length; i++) {
        result.push(Math.exp(x[i]) / sum);
    }

    return result;
}

function normalize(x) {
    let sum = 0;
    for (let i = 0; i < x.length; i++) {
        sum += x[i];
    }

    let result = [];
    for (let i = 0; i < x.length ;i++) {
        result.push(x[i] / sum);
    }

    return result;
}

function apply(x, fn) {
    let matrix = [];
    for (let row = 0; row < x.length; row++){
        let result_row = [];
        for (let col = 0; col < x[0].length; col++){
            let val = fn(x[row][col], row, col);
            result_row.push(val);
        }

        matrix.push(result_row);
    }

    return matrix;
}

function applyRow(x, fn) {
    let matrix = [];

    for (let row = 0; row < x.length; row++) {
        matrix.push(
            fn(x[row]) 
        );
    }

    return matrix;
}

function transpose(x) {
    let matrix = [];
    for (let col = 0; col < x[0].length; col++) {
        let new_row = [];
        for (let row= 0; row < x.length; row++) {
            new_row.push(x[row][col]);
        }

        matrix.push(new_row);
    }

    return matrix;
}

function norm(x) {
    let sum = 0;
    for (let i = 0; i < x.length; i++) {
        sum += Math.pow(x[i], 2);
    }

    return [Math.sqrt(sum)];
}

function arrayRotate(arr, reverse) {
    arr = JSON.parse(JSON.stringify(arr));
    if (reverse) arr.unshift(arr.pop());
    else arr.push(arr.shift());
    return arr;
  }

function shiftVector(interpolated_weight, shift) {
    // interpolated_weight -> M x 1

    left_shift = arrayRotate(interpolated_weight);
    right_shift = arrayRotate(interpolated_weight, true);

    return apply(interpolated_weight, (x, i, j) => {
        return shift[0][1] * x + shift[0][2] * right_shift[i][j] + shift[0][0] * left_shift[i][j];
    })
}

function getShape(x) {
    return [x.length, x[0].length];
}

function concatenate(a, b, dim=0) {
    let result = [];

    if (dim == 0) {
        return a.concat(b);
    } else if (dim == 1) {
       for (let i = 0; i < a.length; i++) {
           let row = [];
           for (let j = 0; j < a[0].length; j++) {
               row.push(a[i][j]);
           }

           for (let j = 0; j < b[0].length; j++) {
               row.push(b[i][j]);
           }

           result.push(row);
       } 

       return result;
    }
}

function produceWeightings(controller_hidden, memory, prev_weight, params, prefix) {
    key_vector = linear(controller_hidden, params, prefix + '.hidden->key');
    strengthen = linear(controller_hidden, params, prefix + '.hidden->strengthen');
    interpolation = linear(controller_hidden, params, prefix + '.hidden->interpolation');
    shift = linear(controller_hidden, params, prefix + '.hidden->shift');
    sharpen = linear(controller_hidden, params, prefix + '.hidden->sharpen');

    // strengthen -> 1 x 1
    strengthen = apply(strengthen, softplus);
    sharpen = apply(apply(sharpen, softplus), (x) => x + 1)

    interpolation = apply(interpolation, sigmoid);

    shift = applyRow(shift, softmax_);

    
    // key_vector -> 1 x N
    // memory -> M x N

    // dot -> M x 1
    dot = matmul(memory, transpose(key_vector));

    // memory_magnitudes -> M x 1
    memory_magnitudes = applyRow(memory, norm); 

    // key_magnitude -> 1
    key_magnitude = norm(key_vector[0]);
    prod_magnitudes = apply(memory_magnitudes, (x) => x * key_magnitude[0]);

    cosine_similarity = apply(dot, (x, i, j) => x / (prod_magnitudes[i][j] + 1e-8));

    cosine_similarity = apply(cosine_similarity, (x) => x * strengthen[0][0]);

    preliminary_weight = transpose(applyRow(transpose(cosine_similarity), softmax_));

    //console.log(getShape(preliminary_weight), '|', getShape(prev_weight));
    
    interpolated_weight = apply(preliminary_weight, function (x, i, j) {
        return interpolation[0][0] * x + (1 - interpolation[0][0]) * prev_weight[j][i];
    });

    //console.log(shift, 'shift vector');
    shifted_weight = shiftVector(interpolated_weight, shift);

    sharpened_weight = apply(shifted_weight, (x) => Math.pow(x, sharpen[0][0]));
    weight_vector = applyRow(transpose(sharpened_weight), normalize);

    return [transpose(weight_vector), key_vector, strengthen, interpolation, shift, sharpen];
}

function getEraseWrite(controller_hidden, memory, prev_weight, params, prefix) {
    let result = produceWeightings(controller_hidden, memory, prev_weight, params, prefix);

    let weight_vector = result[0];

    erase_vector = linear(controller_hidden, params, prefix + '.hidden->erase');
    erase_vector = apply(erase_vector, sigmoid);
    write_vector = linear(controller_hidden, params, prefix + '.hidden->write');
    write_vector = apply(write_vector, Math.tanh);

    return [weight_vector, erase_vector, write_vector, result];
}

function performWrite(memory, weight_vector, erase_vector, write_vector) {
    // weight_vector -> M x 1
    // erase_vector -> 1 x N
    erase_matrix = matmul(weight_vector, erase_vector)
    erase_matrix = apply(erase_matrix, (x) => 1 - x);
    memory = apply(memory, (x, i, j) => x * erase_matrix[i][j]);


    // write
    write_matrix = matmul(weight_vector, write_vector)
    memory = apply(memory, (x, i, j) => x + write_matrix[i][j]);

    return [weight_vector, memory];
}

function read(controller_hidden, memory, prev_weight, params, prefix) {
    let result = produceWeightings(controller_hidden, memory, prev_weight, params, prefix);
    let weight_vector = result[0];
    // weight_vector -> M x 1
    // memory -> M x N

    // output -> N x 1
    return [weight_vector, matmul(transpose(memory), weight_vector), result];
}

function controllerForward(inp, params, prefix) {
    let x = linear(inp, params, prefix + '.l1');
    x = apply(x, (x) => Math.max(0, x));
    let out = linear(x, params, prefix + '.out'); 

    x = linear(x, params, prefix + '.l2');
    x = apply(x, (x) => Math.max(0, x));

    return [x, out];
}

function ntmForward(external_input, memory, read_vector, prev_weights, params, read_heads=1, write_heads=1) {
    let inp = concatenate(external_input, read_vector, dim=1);

    const controller_results =  controllerForward(inp, params, prefix='controller');
    const controller_hidden = controller_results[0];
    const controller_output = controller_results[1];

    let weights = [];
    let read_vector_intermediate = [];
    let other_datas = [];

    let idx = 0;
    for (let head = 0; head < read_heads; head++) {
        const r = read(controller_hidden, memory, prev_weights[idx], params, `read_heads.${head}.layers`);

        const weight_vector = r[0];
        const read_vector = r[1];
        const other_data = r[2];

        weights.push(transpose(weight_vector));
        //console.log(weight_vector, 'read');

        read_vector_intermediate = concatenate(read_vector_intermediate, read_vector, dim=0);

        other_datas.push(other_data);
        idx++;
    }

    let erase_vectors = [];
    let write_vectors = [];
    for (let head = 0; head < write_heads; head++) {
        const w = getEraseWrite(controller_hidden, memory, prev_weights[idx], params, `write_heads.${head}.layers`);

        const weight_vector = w[0];
        const erase_vector = w[1];
        const write_vector = w[2];
        const other_data = w[3];

        erase_vectors.push(erase_vector);
        write_vectors.push(write_vector);

        weights.push(transpose(weight_vector));
        other_datas.push(other_data);

        idx++;
    }

    for (let head = 0; head < write_heads; head++) { 
        const w = performWrite(memory, transpose(weights[read_heads + head]), erase_vectors[head], write_vectors[head]);
        memory = w[1];
    }

    // N x 1 -> 1 x N
    read_vector = transpose(read_vector_intermediate);

    return [memory, read_vector, weights, controller_output, other_datas];
}

function generateMatrix(a, b) {
    let res = [];
    for (let i = 0; i < a; i++) {
        let row = [];
        for (let j = 0; j < b; j++) { 
            row.push(0.);
        }

        res.push(row);
    }

    return res;
}

/*
let read_vector = generateMatrix(1, 10);
let external_input = generateMatrix(1, 7);

let prev_weights = [generateMatrix(1, 40), generateMatrix(1, 40)];
prev_weights[0][0][0] = 1.;
prev_weights[1][0][0] = 1.;

let memory = generateMatrix(40, 10);

for (let i = 0; i < 10; i++) {
    external_input = [[]];
    for (let j = 0; j < 6; j++) {
        external_input[0].push((Math.random() > 0.5) * 1.);
    }

    external_input[0].push(0);

    let result = ntmForward(external_input, memory, read_vector, prev_weights, params.params);
    memory = result[0];
    read_vector = result[1];
    prev_weights = result[2];

    console.log(external_input + '', apply(result[3], (x) => (x > 0.7 ) * 1.)[0] + "");
}


external_input = [[0., 0., 0., 0., 0., 0., 1.]];
let result = ntmForward(external_input, memory, read_vector, prev_weights, params.params);
memory = result[0];
read_vector = result[1];
prev_weights = result[2];

for (let i = 0; i < 10; i++) {
    external_input = [[0., 0., 0., 0., 0., 0., 0.,]];

    let result = ntmForward(external_input, memory, read_vector, prev_weights, params.params);
    memory = result[0];
    read_vector = result[1];
    prev_weights = result[2];

    console.log(external_input + '', apply(result[3], (x) => (x > 0.7) * 1.)[0] + "");
}
*/