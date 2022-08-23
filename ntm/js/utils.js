const e = 2.71;

const getOffset = (e, obj) => {
    const parentOffset = $(obj).offset();

    return [e.pageX - parentOffset.left, e.pageY - parentOffset.top]; 
}    

const between = (x, min, max) => {
    if (x < min || x > max) return false;
    return true;
}

const enforceBounds = (x) => {
    if (x < 0) {
        return 0;
    } else if (x > 1){
        return 1;
    } else {
        return x;
    }
}

const interpolateLinearly = (x, values) => {
    // split values into four lists
    var x_values = [];
    var r_values = [];
    var g_values = [];
    var b_values = [];
    for (i in values) {
        x_values.push(values[i][0]);
        r_values.push(values[i][1][0]);
        g_values.push(values[i][1][1]);
        b_values.push(values[i][1][2]);
    }
    var i = 1;
    while (x_values[i] < x) {
        i = i+1;
    }
    i = i-1;
    var width = Math.abs(x_values[i] - x_values[i+1]);
    var scaling_factor = (x - x_values[i]) / width;
    // Get the new color values though interpolation
    var r = r_values[i] + scaling_factor * (r_values[i+1] - r_values[i])
    var g = g_values[i] + scaling_factor * (g_values[i+1] - g_values[i])
    var b = b_values[i] + scaling_factor * (b_values[i+1] - b_values[i])
    return [enforceBounds(r), enforceBounds(g), enforceBounds(b)];
}

const getColor = (data, negative, interpolate=undefined) => {
    if (negative) {
        data += 1;
        data /= 2;
    }

    let r; 
    let g; 
    let b;
    if (interpolate) {
        r = interpolate[0] * data + 255 * (1 - data)
        g = interpolate[1] * data + 255 * (1 - data)
        b = interpolate[2] * data + 255 * (1 - data) 
    } else {
        let color = interpolateLinearly(data, Greys);
        r = Math.round(color[0] * 255);
        g = Math.round(color[1] * 255);
        b = Math.round(color[2] * 255);
    }

    return 'rgb(' + r + ',' + g + ',' + b + ')';
}

const softmax = (x, strengthen) => {
    if (!strengthen) strengthen = 1;

    let sum = 0;
    for (let i = 0; i < x.length; i++) {
        sum += Math.pow(e, x[i] * strengthen);
    }

    let result = [];
    for (let i = 0; i < x.length; i++) {
        result.push(Math.pow(e, x[i] * strengthen)/sum);
    }

    return result;
}

const dotProd = (v1, v2) => {
    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
        sum += v1[i] * v2[i];
    }

    return sum; 
}

const magnitude = (vector) => {
    let sum = 0;

    for (let i = 0; i < vector.length; i++) {
        sum += Math.pow(vector[i], 2);
    }

    return Math.pow(sum, 0.5);
}
