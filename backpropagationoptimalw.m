clc
clear all
%Dadas las entradas para XOR1
entradas=[0 0;
          0 1;
          1 0;
          1 1];
%Las correspondientes salidas para XOR1
Salidas=[0;1;1;0];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%Dadas las entradas para XOR2
% 
% entradas=[1 1;
%           1 0;
%           0 1;
%           0 0];
% %Las correspondientes salidas para XOR2
% Salidas=[0;1;1;0];
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %Dadas las entradas para XOR3
% % 
% entradas=[0 0;
%           1 1;
%           0 1;
%           1 0];
% %Las correspondientes salidas para XOR2
% Salidas=[0;0;1;1];





%Inicializacion de pesos W
w1=-8.991310050855532;
w2=-8.862225298620457;
w3=3.626816886749211;
%Declaracion del Bias
Bias=1;
%Inicialización de los pesos V
v1=-9.448550660276922;
v2_h=-19.761156661887460;
v3=-9.319183324931240;
v4=14.172626047795450;
%coeficiente de aprendizaje
etha=.1;
%alpha debe ser menor que etha
alpha=0.09;

%Definir numero de epocas maximas
max_Epocs=100000;
%Definir la tolerania en el error
tol=0.01;
%definir contador de epocas
epocas=1;
e_k=ones(4,1);
%inicializar deltak_old h_j_old para convergencia en v
h_j_old=0;
deltak_old=0;
I_v1_old=0;
I_v2_old=0;
I_v3_old=0;
I_v4_old=0;

%inicializar deltak_old h_j_old para convergencia en w
Delta_j_old=0;
I_w1_old=0;
I_w2_old=0;
I_w3_old=0;

%inicializar el error_graph para graficar cada 1000 muestras
cont2=1;
cont_graph=1;
while norm(e_k(1,1))>=tol || norm(e_k(2,1))>=tol || norm(e_k(3,1))>=tol || norm(e_k(4,1))>=tol && epocas<max_Epocs

for c=1:size(entradas,1)
%%%Primer paso: sumar los pesos w multiplicados por las entradas mas bias
S_j=(w1*entradas(c,1))+(w2*entradas(c,2))+Bias*w3;
%Se pasa el valor S_j por la funcion sigma para obtener la salida en esa
%neurona
h_j=1/(1+exp(-S_j));

%%%Para los pesos V se realiza el mismo proceso que en el primer paso
r_k=(v1*entradas(c,1))+(v2_h*h_j)+(v3*entradas(c,2))+Bias*v4;
%la salida se calcula con la funcion sigma
O_k(c,1)=1/(1+exp(-r_k));

%Obtenemos el error de acuerdo a la salia
e_k(c,1)=Salidas(c,1)-O_k(c,1);

%%%Para actulizar los pesos V, definimos a delta_k resultante de las
%%%derivadas parciales del descenso de gradiente
deltak=e_k(c,1)*O_k(c,1)*(1-O_k(c,1));
%ahora definimos el gradiente de Ep=-(-deltak*h_j)
grad_Ep=deltak*h_j;

%%%Para actualizar los pesos W, obtenemos Delta_j, debido a que j=1 neurona solo hay una sola  
Delta_j=deltak*(v1+v2_h+v3+v4)*h_j*(1-h_j);
%Terminos para mejor convergencia para v

I_v1=etha*deltak_old*entradas(c,1)+alpha*I_v1_old;
I_v2=etha*deltak_old*h_j_old+alpha*I_v2_old;
I_v3=etha*deltak_old*entradas(c,2)+alpha*I_v3_old;
I_v4=etha*deltak_old*Bias+alpha*I_v4_old;

%Terminos para mejor convergencia para w
I_w1=etha*Delta_j_old*entradas(c,1)+alpha*I_w1_old;
I_w2=etha*Delta_j_old*entradas(c,2)+alpha*I_w2_old;
I_w3=etha*Delta_j_old*Bias+alpha*I_w3_old;


%Actualización de los pesos v
v1=v1+etha*deltak*entradas(c,1)+alpha*I_v1;
v2_h=v2_h+etha*grad_Ep+alpha*I_v2;
v3=v3+etha*deltak*entradas(c,2)+alpha*I_v3;
v4=v4+etha*deltak*Bias+alpha*I_v4;
%Actualización de los pesos w
w1=w1+etha*Delta_j*entradas(c,1)+alpha*I_w1;
w2=w2+etha*Delta_j*entradas(c,2)+alpha*I_w2;
w3=w3+etha*Delta_j*Bias+ alpha*I_w3;

% Actualizacion de terminos de convergencia para v
deltak_old=deltak;
h_j_old=h_j;
I_v1_old=I_v1;
I_v2_old=I_v2;
I_v3_old=I_v3;
I_v4_old=I_v4;
%Actualización de términos de convergencia para w
Delta_j_old=Delta_j;
I_w1_old=I_w1;
I_w2_old=I_w2;
I_w3_old=I_w3;
end
epocas=epocas+1;
cont2=cont2+1;

if cont2==1000
    graph_error(:,cont_graph)=e_k;
    cont_graph=cont_graph+1;
    cont2=1;
end
end
display(e_k);
epocas_1=epocas/1000;
epocs=1:epocas_1;
plot(epocs*1000,graph_error(:,epocs));

 