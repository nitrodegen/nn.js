
/*

    Very basic and stupid ML library.
    Zero google expirience.
    Everything by myself.

*/

/*** Implementation ***/ 



/* Weights */

const { max } = require("bn.js");
const { error, assert } = require("console");
const fs = require("fs");
const { exit } = require("process");
const DEBUG = 0
const ldash = require('lodash'); 

function randomInRange(min,max) {
 var ret;
 for (;;) {
  ret = min + ( Math.random() * (max-min) * 1.1 );
  if ( ret <= max ) { break; }
 }
 return ret;
}

function argmax(x){

    var big_num = Math.max(...x)
    for(var i = 0 ;i < x.length;i++){
        if(x[i] == big_num){
            return i ;
        }
    }

}
function mat_subtract(x,y){
    
    var out = ldash.cloneDeep(x)
    var idx = 0 ;    

    


    for(var i = 0; i < x.length;i++){

        for(var j = 0 ;j < x[i].length;j++){
            
            out[i][j] =  ( x[i][j] - y[i][j] ) 
        
        }

    }
    return out 

}

function mat_eye(x){

    var out = [] 
    var d = 0 ;
    for(var i = 0 ;i < x;i++){
        var z = new Array(x).fill(0)
        z[i] = 1
        out[d++] = z
    }
   
    return out 

}

function init_weights(x,y){

    var n = x 

    var lower = -(1.0/Math.sqrt(n))
    var upper = Math.abs(lower);


    var z = Array(y).fill(Array(x).fill(0));
    for(var i  =0 ;i <y;i++){
        for(var j = 0;j<x;j++){
           z[i][j]= randomInRange(lower,upper);
        }
    }
        
    return z;

}

/* Layers */ 

/* Reading WASM files for backend */

var mm_module = require('./imps/mm/mm');
const { type } = require("os");
const { NOTIMP } = require("dns");
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

/* Guide: this function is used to set activation formulas on NN layers.
    Options:
        sigmoid,
        relu   
*/

function relu(x){
    return Math.max(0,x);
}


function activate(x,activation){


    for(var i = 0 ;i < x.y;i++){

        for(var j = 0; j < x.x;j++){

            if(activation == "sigmoid"){
                x.matrix[i][j] =  sigmoid_formula(x.matrix[i][j]); 
            }
            if(activation == "relu"){
                x.matrix[i][j] =  relu(x.matrix[i][j]);
            }


        }
    }

    return x;

}

class Sigmoid{

    constructor(){

     
        this.is = undefined
        this.os = undefined


    }
    forward(arg){
 
        this.os = arg.x
        this.is = arg.x;
        
        return activate(arg,"sigmoid");

    }
    partial_backward(x){
        
        var z = ldash.cloneDeep(x)

        for(var i = 0 ;i < x.y;i++){

            for(var j = 0; j < x.x;j++){

              
                z.matrix[i][j] =  z.matrix[i][j]*(1-z.matrix[i][j]);
                
            
            }
        }
        return z

    }


};

class ReLU{

    constructor(){

        this.is = undefined
        this.os = undefined

     

    }
    forward(arg){

        this.os = arg.x;
        this.is = arg.x;
        
        return activate(arg,"relu");

    }

    partial_backward(x){
 
        var z = ldash.cloneDeep(x)

        for(var i = 0; i <x.y;i++){
            for(var j = 0 ; j < x.x;j++){
                var n = z.matrix[i][j]
                if(n > 0){
                    z.matrix[i][j] =1
                }
                if(n <=0 ){
                    z.matrix[i][j] = 0 
                }
            }
        }
        return z;

    }

};



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
 
        console.log(this.y);
        if(this.y >  1){

            
            for(var i = 0 ;i < this.y;i++){
                process.stdout.write("[");

                for(var j = 0; j<this.x;j++){
    //                console.log(this.matrix[j])              
                    process.stdout.write(" "+this.matrix[i][j].toString()+" ");

                }

                process.stdout.write("]");
                process.stdout.write("\n");

            }
            
        }
        if(this.y == 1){
            
            
            process.stdout.write("[");


            for(var i = 0;i <this.x;i++){

                process.stdout.write(" "+this.matrix[i].toString()+" ");
                    
            }
            
            process.stdout.write("]");
            process.stdout.write("\n");


        }
      

        
    }

    transpose(){
 
 
        var aout = [] 
 
        var idx = 0 ;
        for(var i = 0;i < this.x;i++){
            var batch = [] 
            var idy = 0 
            for(var z = 0 ;z < this.y;z++){
                
                batch[idy++] =this.matrix[z][i]
            }
            aout[idx++ ] = batch
        
        }
       
        var temp = ldash.cloneDeep(this.x)

        this.x = this.y
        this.y = temp 

        this.matrix= ldash.cloneDeep(aout)

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
function map_2d_1d(x){
    
    var out = [] 
    var idx = 0 ;;

    for(var i = 0 ;i < x.length;i++){
        if(typeof x[i] == "object"){
         for(var j  = 0; j < x[i].length;j++)
            {
                out[idx++ ] = x[i][j];
            }
        }
        else{
            out[idx++] = x[i];
        }
    }
    return out ;

}
function mat_prod(x,y){
    
    var out = ldash.cloneDeep(x);

    if(y.length == 1 && y[0].length == 1){
        var zd = new Array(x.length).fill(new Array(x[0].length).fill(y[0][0]))
        y = ldash.cloneDeep(zd)
    }

    for(var i = 0 ;i < x.length;i++){
        for(var j  = 0 ; j < x[i].length;j++){
            
            out[i][j] = x[i][j] * y[i][j]
 
        }
    }

    return out;

}

function dot_prod_backend(x,y){

    //console.log("**** dot prod *****")


    var size_x = [x.x,x.y]
    
    var size_y = [y.x , y.y]

    var mat_x = x.matrix;
    var mat_y = y.matrix;
    
    mat_x = remap_mat(mat_x,size_x[0],size_x[1]);
 
    mat_y = remap_mat(mat_y,size_y[0],size_y[1]);
 
    /*
        
        double* mat = (double*)malloc(sizeof(double)*x*y);
        memset(mat,0x00,sizeof(double)*x*y);
        return (void*)mat;

    */
    var x_mod = mm_module._init_mat(size_x[0],size_x[1]);

    mm_module.HEAPF64.set(new Float64Array(mat_x),x_mod/8);

    var x_bck = mm_module.HEAPF64.subarray(x_mod/8,((x_mod/8)+(x.y*x.x)))


    var y_mod = mm_module._init_mat(size_y[0],size_y[1]);

    mm_module.HEAPF64.set(new Float64Array(mat_y),y_mod/8);

    var y_bck = mm_module.HEAPF64.subarray(y_mod/8,((y_mod/8)+(y.y*y.x)))

 
 
    var lin_layer = mm_module.cwrap('dot_prod_single','number',['typedArray','typedArray','number','number', 'number','number'])

    

    var out = lin_layer(x_mod,y_mod,size_x[0],size_x[1],size_y[0],size_y[1])
 
    var addr = out 
    out  = mm_module.HEAPF64.subarray(out/8,(out/8)+(size_x[1]*size_y[1]));
    
    out = ldash.cloneDeep(out);

    var dot =  [[].slice.call(out)];
    
    mm_module._free(x_mod);
    mm_module._free(y_mod);
    mm_module._free(addr);


    return dot;

}


function compute_backend(x,y,bias,yx,yy){
    //console.log(mm_module);
 
    var x_size = [x.x,x.y];
    var y_size = [yx,yy]


    // map 2d to 1d array.
    var a = x.matrix;
    var w = y;
    

    w = remap_mat(w,y_size[0],y_size[1]);
    if(x_size[1] == 1){
        a=a[0]
    }
    var a_be = mm_module._init_mat(x_size[0],x_size[1]);
 
    mm_module.HEAPF64.set(new Float64Array(a),a_be/8);

    var amem = mm_module.HEAPF64.subarray(a_be/8,((a_be/8)+(x_size[0]*x_size[1]) ) );



    var w_be = mm_module._init_mat(y_size[0],y_size[1]);
    mm_module.HEAPF64.set(new Float64Array(w),w_be/8);
    var wmem = mm_module.HEAPF64.subarray(w_be/8,(w_be/8)+(y_size[0]*y_size[1]));


    var b_be = mm_module._init_mat(bias.length,1);
    mm_module.HEAPF64.set(new Float64Array(bias),b_be/8);
    var bmem = mm_module.HEAPF64.subarray(b_be/8,(b_be/8)+bias.length);

    var lin_layer = mm_module.cwrap('compute_linear_layer','number',['typedArray','typedArray','typedArray','number','number','number','number'])


    var out = lin_layer(a_be,w_be,b_be,x_size[0],x_size[1],y_size[0],y_size[1]);

    var out_addr= out;

    out = mm_module.HEAPF64.subarray(out/8,(out/8)+(x_size[1]*y_size[1]));
 
    out = reshape_array(out,[y_size[1],x_size[1]])
    
    mm_module._free(a_be);
    mm_module._free(w_be);
    mm_module._free(b_be);
    mm_module._free(out_addr);
    
    return out;
    
}




function load_static_data(x){


    var dat = fs.readFileSync(x).toString().split("\n");

    var x_arr = new Array(dat.length-1);
    var loaded = 0 

    for(var i = 0;i < dat.length;i++){
    
        

        if(dat[i].length > 1){
            var lin = dat[i].split(",");
                 
            var xnd = new Array(lin.length);

            for(var j = 0;j<lin.length;j++){

                if(lin[j] != ""){
                xnd[j] = parseFloat(lin[j]);
                }
            }
            
            
            if(xnd.length >= 1){


                x_arr[i] = xnd;

                loaded++;
            }

        }
   }   


    return x_arr;   
}



class Linear{

    constructor(input_size,output_size){
    
        
        this.is = input_size;
        this.os = output_size;
        this.output_forward = undefined

        if(!DEBUG){

            this.weights = init_weights(this.is,this.os);
            this.bias = init_weights(this.os,1)[0];
          
        } 
        else{

            this.weights = load_static_data("./weights.dat")
            this.bias = load_static_data("./bias.dat")
            
        }

    
        
    }
    forward(input_matrix){
      
        if(input_matrix.y >1 ){
            
           
            input_matrix.matrix  = remap_mat(input_matrix.matrix,input_matrix.x,input_matrix.y);
            
            input_matrix.y = 1
            
            input_matrix.x = input_matrix.x*input_matrix.y;

        }   

      
 
        var out = compute_backend(input_matrix,this.weights,this.bias,this.is,this.os);        
        var out_tensor = new Tensor(this.os,input_matrix.y);
                
        out_tensor.matrix = out;

        return out_tensor

        
    }




};

//super() in inheritance is used to call the parent constructor function to set its arguments ( if they are needed ) 

/* activations */ 
//https://gist.github.com/yang-zhang/217dcc6ae9171d7a46ce42e215c1fee0


function sum_array(x){
 
    if(x.length  > 0 ){ 
 
        var out = undefined
        
        if( ( typeof x[0] )  === "object"){
            out = new Array(x.length).fill(0)
            
        }
        if(( typeof x[0] ) === "number"){
            out = 0 ;
        }   
        var indexer = 0 ;

        for(var i = 0;i < x.length;i++){
            
            var type = typeof x[i]
            if(type == "object"){
                

                for(var j = 0;j < x[i].length;j++){
 
                    out[indexer] += x[i][j]
                    
                }
 
                indexer++
            
            }
            if(type == "number"){
                out+=x[i]
            }          
            
        }

    }
    return out 

};

function softmax(x){

    for(var i = 0;i < x.length;i++){
        if(typeof x[i] == "object"){
            for(var j = 0 ;j < x[i].length;j++){
                x[i][j] = Math.exp(x[i][j])
            }
        }
        if(typeof x[i] == "number"){
            x[i] = Math.exp(x[i])
        }
    }

    var exp_dat = x
    var  mzd = sum_array(exp_dat)

    for(var i = 0;i < x.length;i++){
        if(typeof x[i] == "object"){
            for(var j = 0 ;j < x[i].length;j++){
                x[i][j] = x[i][j] / mzd
            }
        }
        if(typeof x[i] == "number"){
            x[i] = x[i] / mzd
        }
    }

    return x    

}
function log_softmax(x){

    for(var i = 0;i < x.length;i++){
        if(typeof x[i] == "object"){
            for(var j = 0 ;j < x[i].length;j++){
                x[i][j] = Math.exp(x[i][j])
            }
        }
        if(typeof x[i] == "number"){
            x[i] = Math.exp(x[i])
        }
    }targ.matrix[0][0]

    var exp_dat = x
    var  mzd = sum_array(exp_dat)

    for(var i = 0;i < x.length;i++){
        if(typeof x[i] == "object"){
            for(var j = 0 ;j < x[i].length;j++){
                x[i][j] = Math.log ( x[i][j] / mzd ) 
            }
        }
        if(typeof x[i] == "number"){
            x[i] = Math.log( x[i] / mzd )
        }
    }

    return x    

}

function nll(input,target){

   
    console.log(input.length,target.length)
    var output = new Array(input.length*target.length).fill(0)
    var idx = 0 
    for(var i = 0;i  < input.length;i++){
        for(var j = 0;j < target.length;j++){ 
            output[idx++] = -input[i][target[j]]
 
        }
    }

    var sumz = sum_array(output)
    return sumz / output.length

}   

class LossOutput{
    //output_loss , predicted_data , real_data
    constructor(ol,pd,rd,name,bck){
        
        this.output_loss = ol
        this.predicted_data = pd
        this.real_data = rd
        this.name = name 
        this.partial_backward = bck
        
    }

}

class CrossEntropyLoss extends Function{
    constructor(){
 
        super("...args","return this.bar(...args)")
        
        return this.bind(this);
        
    }


    bar(pred,real){

     
        assert(pred.y === real.y);

        var matp = pred.matrix;
        var mat_real = real.matrix;
  
        var log_pred = log_softmax(matp)

        var out = nll(log_pred,mat_real)

        var lout = new LossOutput(out,pred,real,this.constructor.name,undefined);


        return lout

        
    }
};

function MSELoss_backward(pred,targ){
      
    var matp = pred.matrix;
        
    var mat_real = targ.matrix;


    var out = [] 
    var idx = 0;
 
    for(var i = 0 ;i < pred.y;i++){
        for(var z = 0 ; z < targ.x;z++){
            
            for(var j = 0; j < pred.x;j++){
            
                var o = matp[i][j]
                var l = mat_real[i][z]
                // ?? 
                out[idx++] = l-o

            }
        }
    }

   
    var lout  = new LossOutput(out,pred,targ,this.constructor.name);
    return lout;


}

class MSELoss extends Function{
    constructor(){
 
        super("...args","return this.bar(...args)")
        
        return this.bind(this);
        
    }


    bar(pred,real){

     
        assert(pred.y === real.y);
        
        var matp = pred.matrix;
        var mat_real = real.matrix;
  
        if(DEBUG){

            matp = [[-0.0248,  0.0578,  0.1540,  0.1785, -0.2152, -0.1444, -0.2141,  0.1090,
          0.0058,  0.0099]]
            mat_real = [[1,2,3]]
           
        }

        var out = 0

        for(var i = 0 ;i < pred.y;i++){
            for(var z = 0 ; z < real.x;z++){
                
                for(var j = 0; j < pred.x;j++){
                   
                    var o = matp[i][j]
                    var l = mat_real[i][z]
                    out+=(Math.pow((l - o ),2)/pred.x)

                }
            }
        }
 
        var lout = new LossOutput(out,pred,real,this.constructor.name,this.partial_backward);


        return lout

        
    }
    partial_backward(pred,targ){

        return MSELoss_backward(pred,targ);

    }

};

/* datasets */ 

class MNIST{

    
    constructor(){

        this.images = new Array;

    }

 
    async load_net_db(url){
        
        if(url == null){
            url = "http://localhost:8000/train.csv";
        }
    
        var reqz = await fetch(url).then(res=>res.blob()).then(data=>{
            console.log(data);
        })
        

        console.log(reqz);

    }
     load_local_csv(fname,nimgs=0xff){
        
        var f =  fs.readFileSync(fname);
     

        f= f.toString();
        f=f.split("\n");
        
        if(nimgs == 0xff){
            nimgs = f.length;
        }
        
        this.images = new Array(nimgs);
        var licnt = 0;  

        for(var i = 1; i < nimgs;i++){
            
            var img = f[i].split(",");
 
            var label = parseInt(img[0]);
            
            img.shift();

            var minval = 0.00
            var maxval = 255.00
            // loading + normalizing 

            var image = new Array(28).fill(new Array(28).fill(0));
            
            for(var j  = 0 ;j < 28;j++){
                var md = new Array(28).fill(0); 
                for(var z = 0; z < 28; z ++ ){
                    md[z] = ((parseInt(img[28*j+z])-minval)/(maxval-minval))
                }
                image[j] = md;
            }
            

        
           
            var tn = new Tensor(28,28);
            tn.matrix = image;
            image = tn;
          
            var batch = new Array(2);
            batch[0] = image;
            batch[1] = label;
            //console.log(batch)

            this.images[licnt++] = batch;

        }
        //exit(1);
        return this.images;
         
        

    }





}


class NNClass{

    constructor(lr){        
        
        this.added_layers = []
        
        this.layer_count = 0 
        
        this.parameters = [] 
        
        this.outputs = []

        this.lr = lr 

    }
    //default forward function  , can be overwritten.
    forward(x){
        
        assert( this.layer_count > 0 ) 
        var idx =0 ; 
        var lolz = {} 

        for(var i = 0;i < this.layer_count;i++){

            //this.outputs[i] = x;

            
            if(this.added_layers[i].constructor.name != "LossOutput"){
            x = this.added_layers[i].forward(x)
        //!!!! TODO: **** browser doesnt support cloneDeep **** !!!! 

            const z = ldash.cloneDeep(x)
            this.added_layers[i].output_forward = z
            this.outputs[idx] = z
            
            idx++
            }
            
        }
      
  
        return x

    }
    backward(loss){


        // *** TODO: implement in WASM, for now in vanilla JS 
     
        var output_size = loss.predicted_data.x*loss.predicted_data.y
      
        var j_cost = loss.output_loss;

        var diag_out = mat_eye(output_size)
        var dg = [] 
        for(var i = 0;i < loss.real_data.y;i++){
            for(var j = 0 ;j < loss.real_data.x;j++){
                
                dg.push(diag_out[loss.real_data.matrix[i][j]]);
            }
        }
 
        diag_out = dg 
 
        var e_loss = mat_subtract(this.outputs[this.layer_count-1].matrix,diag_out)
        
        var count_lay = this.layer_count

        var prev_node = e_loss

        var n_nodes =0 ;

        var updated_weights =[] 
        var upd_w_idx = 0; 
      //  console.log(this.added_layers)

        for(var i = this.layer_count-1;i >=0 ; i--){
            
            //chain rule 
            var lname =  this.added_layers[i].constructor.name
        
            // **** TODO : add support for more layers later on 
            
            if(lname === "Linear"){

                var w = this.added_layers[i].weights    
                
              

                var dw1 = []
                var idx = 1;
                dw1[0] = prev_node

                var bdx = i+1;

                while(1){
                    if(bdx >=this.layer_count){
                        break
                    }
                    if(this.added_layers[bdx].constructor.name == "Linear"){
                        break;
                    }
                   // console.log(this.added_layers[bdx].constructor.name)


                    var z =  ldash.cloneDeep(this.outputs[bdx].matrix)

                   
                    z = this.added_layers[bdx].partial_backward(z)
                    
                    dw1[idx++] = z 
                    
                    bdx--;

                }

                if(dw1.length == 2){
        
                    dw1  = mat_prod(dw1[0],dw1[1]);
                
                }
                else{
                    dw1  = dw1[0]
                }
                var d1 = new Tensor(dw1[0].length,dw1.length)
                d1.matrix = ldash.cloneDeep(dw1)
            

                var wd = new Tensor(w[0].length,w.length)
                wd.matrix = ldash.cloneDeep(w)
                wd.transpose()

               // console.log(d1.x,d1.y,wd.x,wd.y)
                var out = dot_prod_backend(d1,wd)
               
                prev_node = ldash.cloneDeep(out);
                updated_weights[upd_w_idx++ ] = ldash.cloneDeep(prev_node)


                n_nodes++;
                                   
               // console.log("***********************************")
           
            }
            
            
        }   
        
        updated_weights = updated_weights.reverse()
        upd_w_idx = 0 ;

        for(var i = 0; i < this.layer_count;i++){
            if(this.added_layers[i].constructor.name == "Linear"){
                
                var w = mat_prod(updated_weights[upd_w_idx],[[this.lr]])

                var w_old = this.added_layers[i].weights
            
                var zod = new Array(w_old.length).fill(w[0])
                
                w = ldash.cloneDeep(zod)


 //               console.log(w_old[0].length,w_old.length)
               
                w_old = mat_subtract(w,w_old)

                this.added_layers[i].weights = ldash.cloneDeep(w_old)
                
                upd_w_idx++
                
                //console.log("**")

            }
        }
        
//        exit(1)
    }


};


class BasicNN extends NNClass{


    constructor(lr){

        super(lr)

        this.added_layers[this.layer_count++] = new Linear(784,256);
        
        this.added_layers[this.layer_count++] = new ReLU();
        
        this.added_layers[this.layer_count++] = new Linear(256,128);
        
        this.added_layers[this.layer_count++] = new Sigmoid()

        this.added_layers[this.layer_count++] = new Linear(128,10);
        



    }

}

/* Tests */
class Test_1{

    constructor(){
        this.nn = new BasicNN(2e-4)
    }
    forward_prop(x){

        var img = x[0]; 

        img.matrix = remap_mat(img.matrix,28,28);
        img.x = 784;
        img.y = 1;
        
        var out = this.nn.forward(img)
        
        return out 
    }

    async run_test(){


        const NIMG = 2048;


        var db = new MNIST();
        var data = db.load_local_csv('./train.csv',NIMG);


        var criterion = new MSELoss()

        const EPOCHS = 300 



        for(var i = 0;i < EPOCHS;i++){
        
        
            for(var j = 0 ;j < NIMG;j++){
                
            
                //this still doenst work ,cant figure out why tho...

                if(db.images[j][0].matrix[0] != null ){
 
                    var targ = new Tensor(1,1)
                                       
                    targ.matrix =  [  [ db.images[j][1] ]   ] 
 
                    var out = this.forward_prop(db.images[j]);
                 
                    var loss = criterion(out,targ);

                    var mg = argmax(out.matrix[0])
                    

                    console.log("loss:",loss.output_loss,mg,targ.matrix[0][0])                    
                    this.nn.backward(loss);

                
                }
               

            }
         
        }

        

    }

};
var t1 = new Test_1();
t1.run_test();

