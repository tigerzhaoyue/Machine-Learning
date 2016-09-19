#include "iostream"
#include "fstream"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
using namespace std;

#define example 1000
#define Neuron 50
#define LearnLim 20000
#define A 0.2
#define B 0.4

double data_in[example][2],data_out[example][1];
double Max_a,Min_a,Max_b,Min_b,Max_out,Min_out;
double w1[Neuron][2],w2[1][Neuron],dw1[Neuron][2],dw2[1][Neuron];
double sigmoid[Neuron];
double output_data,offset;

int calculate(int k){
	double sum,f;
//first layer
	for(int i=0;i<Neuron;i++){	
		sum=0;
		for(int j=0;j<2;j++)
			sum=sum+w1[i][j]*data_in[k][j];
		sigmoid[i]=1/(1+exp(-1*sum));
	}

//second layer
	sum=0;
	for(int j=0;j<Neuron;j++)
			sum=sum+w2[0][j]*sigmoid[j];
	output_data=sum;
	return 0;
}

void Backpropagation(int k){
	double t=0;
	for(int i=0;i<Neuron;i++){
	//second layer	
		t=t+(output_data-data_out[k][0])*w2[0][i];
		dw2[0][i]=A*dw2[0][i]+B*(output_data-data_out[k][0])*sigmoid[i];
		w2[0][i]=w2[0][i]-dw2[0][i];
	//first layer
		for(int j=0;j<2;j++){
			dw1[i][j]=A*dw1[i][j]+B*t*sigmoid[i]*(1-sigmoid[i])*data_in[k][j];
			w1[i][j]=w1[i][j]-dw1[i][j];
		}
	}
}

void Train(){
	int times=0;
	do{
		offset=0;	//error of training
		for(int i=0;i<example;i++){
			calculate(i);
			for(int j=0;j<1;j++)
				offset=offset+fabs((output_data-data_out[i][j])/data_out[i][j]);
			Backpropagation(i);
		}
		printf("%d :%lf\n",times,offset/example);
		times++;

	}while(!(times>LearnLim || offset/example<0.01));
}

void InitExample(){
	float a,b;
	srand((unsigned)time(NULL));
	ofstream in("INPUT.txt");
	if(in.is_open()==0)
		cout<<"ERROR!"<<endl;

	ofstream out("OUTPUT.txt");
	if(out.is_open()==0)
		cout<<"ERROR!"<<endl;

	for(int i=0;i<example;i++){
		a=rand()%1000/100.0000;
		b=rand()%1000/100.0000;
		in<<a<<" "<<b<<endl;
		out<<a+b<<endl;
	}
	in.close();
	out.close();
	return ;
}

void InitBP(){
	Min_a=Max_a=data_in[0][0];
	Min_b=Max_b=data_in[0][1];
	Min_out=Min_out=data_out[0][0];
	for(int i=0;i<example;i++){
		Max_a=Max_a>data_in[i][0]?Max_a:data_in[i][0];
		Min_a=Min_a<data_in[i][0]?Min_a:data_in[i][0];
	}
	for(int i=0;i<example;i++){
		Max_b=Max_b>data_in[i][1]?Max_b:data_in[i][1];
		Min_b=Min_b<data_in[i][1]?Min_b:data_in[i][1];
	}
	for(int i=0;i<example;i++){
		Max_out=Max_out>data_out[i][0]?Max_out:data_out[i][0];
		Min_out=Min_out<data_out[i][0]?Min_out:data_out[i][0];
	}

//regulazilation
	for(int i=0;i<example;i++)
		data_in[i][0]=(data_in[i][0]-Min_a+1)/(Max_a-Min_a+1);
	for(int i=0;i<example;i++)
		data_in[i][1]=(data_in[i][1]-Min_b+1)/(Max_b-Min_b+1);
	for(int i=0;i<example;i++)
		data_out[i][0]=(data_out[i][0]-Min_out+1)/(Max_out-Min_out+1);

//init parameters
	for (int i = 0; i < Neuron; i++)	
		for (int j = 0; j < 2; j++){	
			w1[i][j]=rand()*2.0/RAND_MAX-1;
			dw1[i][j]=0;
		}

	for (int i = 0; i < Neuron; i++)	
		for (int j = 0; j < 1; j++){
			w2[j][i]=rand()*2.0/RAND_MAX-1;
			dw2[j][i]=0;
		}

}
double Neuron_sum(double a,double b)
{
	double sum,y;
	a=(a-Min_a+1)/(Max_a-Min_a+1);
	b=(b-Min_b+1)/(Max_b-Min_b+1);

	for(int i=0;i<Neuron;i++){
		sum=0;
		sum=w1[i][0]*a+w1[i][1]*b;
		sigmoid[i]=1/(1+exp(-1*sum));
	}
	sum=0;
	for(int j=0;j<Neuron;j++){
		sum=sum+w2[0][j]*sigmoid[j];
	}

	return sum*(Max_out-Min_out+1)+Min_out-1;
}
int main(){
	double a,b;
	InitExample();
//Read examples
	FILE *fp1,*fp2;
	if((fp1=fopen("INPUT.txt","r"))==NULL)
		cout<<"ERROR!"<<endl;
	for(int i=0;i<example;i++)
		for(int j=0; j<2; j++)
			fscanf(fp1,"%lf",&data_in[i][j]);
	fclose(fp1);

	if((fp2=fopen("OUTPUT.txt","r"))==NULL)
		cout<<"ERROR!"<<endl;
	for(int i=0;i<example;i++)
		for(int j=0; j<1; j++)
			fscanf(fp1,"%lf",&data_out[i][j]);
	fclose(fp2);


//BP Neuron	
	InitBP();
	Train();
	cout<<"Done!!!"<<endl;
	while(1){
	cout<<"Please input A and B:";
	cin>>a;
	cin>>b;
	cout<<"Neuron_sum is :"<<Neuron_sum(a,b)<<endl;
	}
	return 0;
}
