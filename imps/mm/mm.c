#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


void dbg_arr(double *x,int xz,int yz){
    for(int i = 0; i< yz;i++){
        printf("[");
        for(int j = 0 ;j < xz;j++){
            printf(" %lf ",x[xz*i+j]);
           // printf(" %c ",(char)x[xz*i+j]);
        
        }
        printf("]\n");
    }

}

__attribute__((used)) double* transpose_matrix(double *a ,int X , int Y){


    double *out =(double *)malloc(sizeof(double)*X*Y);
    memset(out,0x00,sizeof(double)*X*Y);


    for(int i = 0 ; i < Y ;i ++ ){
        for(int j = 0 ; j < X ;j++){
            out[Y*j+i] = a[X*i+j];
        }
    }

    return out;




}


double dot_product_1d(double * a, double * b ,int size){ 
    double out = 0;
    for(int i = 0; i < size;i++){
        out +=  a[i] * b[i];
    }
    return out ;
}

double * init_mat(int x,int y){
    
    double *out = (double*)malloc(sizeof(double)*x*y);
    assert(out != NULL);

    memset(out,0x00,sizeof(double)*x*y);
    return out;

}

__attribute__((used)) double* compute_linear_layer(double *a ,double *b , double *bias, int xa , int ya,int xb,int yb ){

    double *out = (double*)malloc(sizeof(double)*xa*yb);
    memset(out,0x00,sizeof(double)*xa*yb);

    int looper = 0; 
    for(int i = 0;i < ya;i++){

        double *a_batch = init_mat(xa,1);
        int al = 0 ;

        for(int j = 0 ;j < xa;j++){
            a_batch[al++] = a[xa*i+j];
        }


        int bias_it = 0 ;


        for(int j = 0; j < yb;j++){

            int bl =0 ; 
            double *b_batch = init_mat(xb,1);
            for(int z = 0; z < xb;z++){
                b_batch[bl++] = b[xb*j+z];
            }
            double out_b = dot_product_1d(a_batch,b_batch,xb) + bias[bias_it++];
            out[looper ++ ] =out_b;
 
            free(b_batch);            

        }

        bias_it = 0x00 ;

        free(a_batch);

    }


    return out;    

}

__attribute__((used)) double* matmul(double *a ,double *b , int xa , int ya,int xb,int yb) {
    double *out = (double*)malloc(sizeof(double)*xa*ya);
    memset(out,0x00,sizeof(double)*xa*ya);

    //mapping mat into 1d array (for speed purposes and memory managment)
    // A[width*row + col]
    // y = row 
    // x = col 


    if(ya !=  1 || yb != 1){

      //  if(xa == xb && ya == yb){
            
            int zk = 0; 
            printf("\n** b arr ** ");
            //double* nd = transpose_matrix(b,xb,yb);
            //dbg_arr(nd,yb,xb);
            //memcpy(b,nd,sizeof(double)*yb*xb);

            for(int i = 0 ;i < ya;i++){
                for(int j = 0 ;j < xa;j++){

                    double zd = 0x00 ;

                    for(int z = 0; z < yb;z++){
                        //a[i][z] * b[z][j]

                    //  printf("\nx:%c y:%c  zk:%d",(char)a[x*i+z],(char)b[y*z+j],zk);
                        double num1 = a[xa*i+z];
                        double num2 = b[xb*z+j];
                        printf("\nx:%lf y:%lf  zk:%d\n",num1,num2,zk);
                        zd+= num1*num2;


                    }  

                    out[zk++] =zd;

                // printf("\nzd:%lf",zd);   
                }
            }
      //  }
        
    }

    else{
        printf("\n??");
        double ox = 0x00;

        for(int i = 0 ;i < ya;i++){
            for(int j = 0 ; j< xa;j++){
                ox+= ( a[xa*i+j]*b[xa*i+j]);
            }
        }
        out[0] = ox;

    }
    return out;
}

int main(){


    double x[4]= {0.1,0.2,0.2,0.3};
    double y[8] = {0.4369,0.4151 ,-0.4103,0.6051 ,0.4435,0.3426 ,-0.0327,-0.0064};

  
    
    //exit(1);
    int sizex = 2;
    int sizey= 2;

    printf("\n ** A ** \n");
    
    dbg_arr(x,2,2);
    printf("\n ** B ** \n");
    
    dbg_arr(y,2,4);

    double bias[4] = {-0.19763727 , -0.5404696  , 0.40338537 ,-0.6749172 };
    double *out = compute_linear_layer(x,y,bias,2,2,2,4);


    printf("\n*** out ** \n");
    dbg_arr(out,2,2);
    free(out);

}
