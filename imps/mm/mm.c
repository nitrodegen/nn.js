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

double * init_mat(int x,int y){
    
    double *out = (double*)malloc(sizeof(double)*x*y);
    assert(out != NULL);

    memset(out,0x00,sizeof(double)*x*y);
    return out;

}

double dot_product_1d(double * a, double * b ,int size){ 
    double out = 0;
    for(int i = 0; i < size;i++){
        
      
          //  printf("a:%lf b:%lf out:%lf\n",a[i],b[i],out);  
        
        out += (  a[i] * b[i] );


    }
    

//    printf("********\n");
    return out ;
}

double * dot_prod_single(double *a, double * b ,int ax,int ay,int bx,int by){ 



    double * out = init_mat(ay,by   );
    int idx = 0  ;

    for(int i = 0 ;i < by;i++){
    
        double batch_b[bx];
        for(int j = 0 ;j < bx;j++){
            batch_b[j] = b[bx*i+j];
        }


        double batch_a[ax];
        for(int j = 0 ;j < ay;j++){
            batch_a[j]= a[ax*i+j];
        }   

        double bout = dot_product_1d(batch_a,batch_b,bx);
        //printf("%lf\n",out);
        out[idx++ ] = bout;



    }

    return out;

}




__attribute__((used)) double* compute_linear_layer(double *a ,double *b , double *bias, int xa , int ya,int xb,int yb ){

    double *out = (double*)malloc(sizeof(double)*ya*yb);
    //printf("xa:%d ya:%d xb:%d yb:%d\n",xa,ya,xb,yb);

    memset(out,0x00,sizeof(double)*ya*yb);

    int looper = 0; 

    for(int i = 0;i < ya;i++){

        double *a_batch = init_mat(xa,1);
        int al = 0 ;
       // printf("\nbatch a size:%d\n",xa);
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

            double out_b = dot_product_1d(a_batch,b_batch,xb)+bias[bias_it++];
            out[looper ++ ] =out_b;
 
            free(b_batch);            

        }

        bias_it = 0x00 ;

        free(a_batch);

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
