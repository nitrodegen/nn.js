
/*

    Very basic and stupid ML library.
    Zero google expirience.
    Everything by myself.

*/

/*** Implementation ***/ 



/* Weights */

const { error } = require("console");
const fs = require("fs");
const { exit } = require("process");

function init_weights(x,y){

    var z = Array(y).fill(Array(x).fill(0));
    for(var i  =0 ;i <y;i++){
        for(var j = 0;j<x;j++){
           z[i][j]= Math.random()*Math.sqrt(x/y);
        }
    }
        
    return z;

}

/* Layers */ 

/* Reading WASM files for backend */

var mm_module = require('./imps/mm/mm');
const DOUBLE_SIZE = 8 ;

function remap_mat(a,sx,sy){

    var arr = new Array(sx*sy);

    var loop =0x00;
    for(var i = 0; i< sy;i++){
        for(var j = 0 ;j <sx;j++){
            arr[loop++] = a[i][j];
        }
    }
    return arr;
}

/* Activations */ 

function sigmoid_formula(x){
    return 1/(1+Math.exp(-x));
}


function sigmoid(x){


    for(var i = 0 ;i < x.y;i++){

        for(var j = 0; j < x.x;j++){

            x.matrix[i][j] =  sigmoid_formula(x.matrix[i][j]);
  
        }
    

    }

    return x;

}


/* Tensor + other stuff */ 

class Tensor{

    constructor(x,y){

        this.x = x;
        this.y = y;
        this.i = 0 ; 
        this.j = 0 ;

        if(this.x <= 0 && this.y  <= 0){
            return undefined;
        }
        
        this.matrix = Array(y);
        for(var i  = 0; i < this.y;i++){
            var newarr = new Array(this.x);
            for(var j = 0 ;j < this.x;j++){
                newarr[j] = 0x00;
            }

            this.matrix[i] = newarr;
        }
        return 1;
  
 
    }
    
    av(value){
        try
        {   

            if(this.matrix[this.i][this.j] != undefined){
                
                //console.log("VALUE:",value,"i:",this.i,"j:",this.j);
                this.matrix[this.i][this.j] = value;
                if(this.j == this.x-1){
                    this.j = 0;
                    this.i++;
                }
                else{
                    this.j++;
                }
            
            }

        }catch{
            console.log("failed val:",this.i,this.j);
        }
        
    }

    init_zero(){
           this.matrix.fill(Array(this.x).fill(0));
    }
    set_all(x){
        this.matrix.fill(Array(this.x).fill(x));
    }

    display(){

        for(var i = 0 ;i < this.y;i++){
            process.stdout.write("[");

            for(var j = 0; j<this.x;j++){
                process.stdout.write(" "+this.matrix[i][j].toString()+" ");
            }

            process.stdout.write("]");
            process.stdout.write("\n");

        }
    
        
    }

};
function reshape_array(x,size){
    
    var out =new  Array(size[1]).fill(Array(size[0]));

    for(var i =0 ;i < size[1];i++){
       
        var md = new Array(size[0]);
       
        for(var j = 0 ;j < size[0];j++){
       
            md[j] = x[size[0]*i+j];
       
        }
        out[i] = md;
    }

    return out;

}

function compute_backend(x,y,bias,yx,yy){
    //console.log(mm_module);


    var x_size = [x.x,x.y];
    var y_size = [yx,yy]

    // map 2d to 1d array.
    var a = x.matrix;
    var w = y;


    a = remap_mat(a,x_size[0],x_size[1]);
    w = remap_mat(w,y_size[0],y_size[1]);

    var a_be = mm_module._init_mat(x_size[0],x_size[1]);
    mm_module.HEAPF64.set(new Float64Array(a),a_be/8);
    var amem = mm_module.HEAPF64.subarray(a_be/8,(a_be/8)+a.length);


    var w_be = mm_module._init_mat(y_size[0],y_size[1]);
    mm_module.HEAPF64.set(new Float64Array(w),w_be/8);
    var wmem = mm_module.HEAPF64.subarray(w_be/8,(w_be/8)+w.length);


    var b_be = mm_module._init_mat(bias.length,1);
    mm_module.HEAPF64.set(new Float64Array(bias),b_be/8);
    var bmem = mm_module.HEAPF64.subarray(b_be/8,(b_be/8)+bias.length);

    var lin_layer = mm_module.cwrap('compute_linear_layer','number',['typedArray','typedArray','typedArray','number','number','number','number'])

    var out = lin_layer(a_be,w_be,b_be,x_size[0],x_size[1],y_size[0],y_size[1]);

    var out_addr= out;

    out = mm_module.HEAPF64.subarray(out/8,(out/8)+(y_size[0]*y_size[1]));
    out = reshape_array(out,[y_size[1],y_size[0]])
    
    

    mm_module._free(a_be);
    mm_module._free(w_be);
    mm_module._free(b_be);
    mm_module._free(out_addr);
    
    return out;
    
}


class Linear{

    constructor(input_size,output_size){
    
        this.is = input_size;
        this.os = output_size;
        this.weights = init_weights(this.is,this.os);
        this.bias = init_weights(this.os,1)[0];
 
        
    }
    exec(input_matrix){

       var out = compute_backend(input_matrix,this.weights,this.bias,this.is,this.os);


       var out_tensor = new Tensor(this.os,this.is);
        out_tensor.matrix = out;
        return out_tensor

    }




};



/* Tests */

function test_1(){

    const W = [
        [
            0.4369,  0.4151
        ],
        [
            -0.4103,  0.6051
        ],
        [
            0.4435,  0.3426
        ],
        [
            -0.0327, -0.0064
        ],
       
    ]
    const BIAS = [ 
        -0.19763727 , -0.5404696  , 0.40338537 ,-0.6749172 
    ]

    var lin = new Linear(2,4);
    lin.weights = W;
    lin.bias= BIAS;


    var x = new Tensor(2,2);
    

    x.init_zero();
    
    var zd = [
        [
            0.1,0.2
        ],
        [
            0.2,0.3
        ]
    ]
    console.log(x.matrix[0][0]);
    x.matrix = zd
 
 
    console.log(x);
    console.log(lin.weights);
 
    console.log("****** exec ******");
    var o = lin.exec(x);
 
    o = sigmoid(o);
    console.log(o)






}


test_1();
