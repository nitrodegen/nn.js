
const Module = require('./mm.js');
const process =require('process')
async function lomx(){


    var x  =3; // size of array ;
    var a = [4,5,69]
    var pointer = Module._malloc(8*x);
    Module.HEAPF64.set(new Float64Array(a), pointer/8);
    var mema = Module.HEAPF64.subarray(pointer/8,pointer/8+x);

    x++;
;
    console.log("array address:","0x"+pointer.toString(16)   );
    console.log(mema);

    var ret_array = Module.cwrap('ret_array','number',['typedArray','number']);
    
    console.log("OUT:",ret_array(pointer,x));
    
    return pointer;



    
    

    

    
}
  lomx();




