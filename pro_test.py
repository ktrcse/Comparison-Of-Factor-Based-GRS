#random users grouping

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# pass in column names for each CSV
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_df= pd.read_csv('ml-100k/u1.base', sep='\t', names=r_cols,
                      encoding='latin-1')
# the movies file contains columns indicating the movie's genres
# let's only load the first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')
Group_comp=[]
mean_accur_AF=[]
mean_accur_BF=[]
mean_accur_WBF=[]
prec_AF=[]
prec_BF=[]
prec_WBF=[]
recal_AF=[]
recal_BF=[]
recal_WBF=[]
F_AF=[]
F_BF=[]
F_WBF=[]
for N in range(2,13):
        Group=[]
        accur_AF=[]
        accur_BF=[]
        accur_WBF=[]
        prec1=[]
        prec2=[]
        prec3=[]
        recall1=[]
        recall2=[]
        recall3=[]
        F1=[]
        F2=[]
        F3=[]        
        for i in range (1,11):
                #grouping
                #print('group size...')
                #print(N)
                vals=np.random.choice(ratings_df['user_id'].unique(), N, replace=False)
                G_df=ratings_df.set_index('user_id').loc[vals].reset_index()

                # Format ratings matrix to be one row per user and one column per movie
                R_df = G_df.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0)
                R_df.head()

                # Normalize by each users mean and convert it from a dataframe to a numpy array
                R = R_df.as_matrix()
                user_ratings_mean = np.mean(R, axis = 1)
                R_demeaned = R - user_ratings_mean.reshape(-1, 1)

                # Singular Value Decomposition (SVD)
                # Scipy and Numpy both have functions to do the SVD. Scipy is used because no. of latent factors can be defined 
                from scipy.sparse.linalg import svds
                
                if(N==2):
                        U, sigma, Vt = svds(R_demeaned, k = 1)
                else:
                        U, sigma, Vt = svds(R_demeaned, k = 2) 
                #print(G_df)
                #print('user factors')
                #print(U)
                #print('Item factors')
                #print(Vt)
                
                #compute user bias
                Ut=np.transpose(U)
                U_bt=sum(Ut)/len(Ut)
                U_b=np.transpose(U_bt)
	
                #compute item bias
                I_b=sum(Vt)/len(Vt)

                #compute muue value
                m_G_df=np.matrix(G_df)
                mmue=np.mean(m_G_df[:,2])
                
                
                #after factorization....................
	
                #group factor
                pg_AF1=sum(U)/len(U)
                #pg_AF1=np.sum(U,axis=0)/np.sum(np.matrix(R_df.gt(0).sum(axis=1)))
                #pg_AF1=np.median(U,axis=0)
                #group bias
                bg_AF1=np.mean(U)
                #bg_AF1= np.sum(U_b)/np.sum(np.matrix(R_df.gt(0).sum(axis=1)))
                #bg_AF1=np.median(U_b)
                '''
                print('group factors from AF')
                print(pg_AF)
                print('group biases from AF')
                print(bg_AF)''' 
	
                #compute BF...................

                #computing A matrix
                z=np.ones((len(Vt[0]),1),dtype=int)
                A=np.append(np.transpose(Vt),z,axis=1)
                
                #computing virtual user rating
                #m=np.matrix(R_df)
                #VR=sum(m)/len(m)
                VR=np.matrix(R_df[R_df > .01].min(axis=0))                  

                #compute virtual user signature
                Sg=[]
                for j in range(0,len(Vt[0])):

                        #item-i bias
                        bi=I_b.item(j)

                        Sg.append(VR.item(j)-mmue-bi)

                lamda=0.055


                #multiply AT * A
                AA=np.dot(A.T,A)
                r=len(AA)

                Id_BF=np.identity(r)
                pg_bg_BF1=np.dot(np.linalg.inv(AA+(lamda*Id_BF)),(A.T))
                S=np.matrix(Sg)
                pg_bg_BF=np.dot(pg_bg_BF1,(S.T))

                bg_BF=pg_bg_BF.item(len(pg_bg_BF)-1)
                pg_BF=np.delete(pg_bg_BF,len(pg_bg_BF)-1)
                '''
                print('group factors from BF')
                print(pg_BF)
                print('group biases from BF')
                print(bg_BF) '''
	
                # compute WBF..................

                #compute W matrix
                c=np.count_nonzero(R,axis=0)
                c_d=c.astype(float)/N
                x=np.matrix(c_d)
                w=np.diag(x.A1)

                #multiply AT *w* A
                AW=np.dot(A.T,w)
                AA1=np.dot(AW,A)

                r1=len(AA1) 

                Id_WBF=np.identity(r1)
                pg_bg_WBF1=np.dot(np.dot(np.linalg.inv(AA1+(lamda*Id_WBF)),A.T),w)
                S_WBF=np.matrix(Sg)
                pg_bg_WBF=np.dot(pg_bg_WBF1,(S_WBF.T))
 
                bg_WBF=pg_bg_WBF.item(len(pg_bg_WBF)-1)
                pg_WBF=np.delete(pg_bg_WBF,len(pg_bg_WBF)-1)
                '''
                print('group factor from WBF')
                print(pg_WBF)
                print('group bias from WBF')
                print(bg_WBF)'''
                
                
                #compute prediction ....................
	                         
                

                #compute AF prediction
                pred_AF1=[]
                pred_AF=[]
                pred_BF=[]
                pred_WBF=[]
                for j in range(0,len(Vt[0])):

                        #item-i bias
                        bi=I_b.item(j)

                        #AF prediction
                        #multiply group factor * item-i factor
                        '''       
                        m_gf_AF = np.matrix(pg_AF)
                        qi_AF=zip(*Vt)[j]
                        m_qi_AF = np.matrix(qi_AF)
                        mm_AF=m_gf_AF*np.transpose(m_qi_AF)
                        m_r_AF=mm_AF.item(0)
                        '''
                        m_gf_AF1 = np.matrix(pg_AF1)
                        qi_AF1=zip(*Vt)[j]
                        m_qi_AF1 = np.matrix(qi_AF1)
                        mm_AF1=m_gf_AF1*np.transpose(m_qi_AF1)
                        m_r_AF1=mm_AF1.item(0)
                        
                        #group predicition
                        #pred_AF.append(int(round(mmue+bi+bg_AF+m_r_AF)))
                        pred_AF1.append(int(round(mmue+bi+bg_AF1+m_r_AF1)))
                        
                        #BF prediction
                           
                        #multiply group factor * item-i factor
                        m_gf_BF = np.matrix(pg_BF)
                        qi_BF=zip(*Vt)[j]
                        m_qi_BF = np.matrix(qi_BF)
                        mm_BF=m_gf_BF*np.transpose(m_qi_BF)
                        m_r_BF=mm_BF.item(0)

                        #group predicition
                        pred_BF.append(int(round(mmue+bi+bg_BF+m_r_BF)))
   
                        #WBF prediction
                           
                        #multiply group factor * item-i factor
                        m_gf_WBF = np.matrix(pg_WBF)
                        qi_WBF=zip(*Vt)[j]
                        m_qi_WBF = np.matrix(qi_WBF)
                        mm_WBF=m_gf_WBF*np.transpose(m_qi_WBF)
                        m_r_WBF=mm_WBF.item(0)

                        #group predicition
                        pred_WBF.append(int(round(mmue+bi+bg_WBF+m_r_WBF)))
   
                #p_AF=np.matrix(pred_AF)
                p_AF1=np.matrix(pred_AF1)
                '''
                print('AF prediction....')
                print(pred_AF)
                print(pred_AF1)'''
                p_BF=np.matrix(pred_BF)
                '''
                print('BF prediction....')
                print(p_BF)'''

                p_WBF=np.matrix(pred_WBF)
                '''
                print('WBF prediction....')
                print(p_WBF)'''
                
                # accuracy.....
                movie=np.matrix(np.unique(G_df.movie_id)).T
                R1_df = G_df.pivot(index = 'movie_id', columns ='user_id', values = 'rating').fillna(0)
                R1=np.matrix(R1_df)
                R[np.where(R==0)]=np.nan
                #R_means=np.nanmean(R,axis=0)
                R_means=np.matrix(np.nansum(R,axis=0)/np.matrix(R_df.gt(0).sum(axis=0))).T
                #R_means=(np.nansum(R,axis=0))/np.count_nanzero(R,axis=0))
                RM_df=pd.DataFrame(R_means)
                a1=np.column_stack((R1,R_means))
                a2=np.column_stack((a1,p_AF1.T))
                a3=np.column_stack((a2,p_BF.T))
                a4=np.column_stack((a3,p_WBF.T))
                acc_AF=np.abs(R_means-p_AF1.T)
                acc_BF=np.abs(R_means-p_BF.T)
                acc_WBF=np.abs(R_means-p_WBF.T)
                accuracy1=np.column_stack((a4,acc_AF))
                accuracy2=np.column_stack((accuracy1,acc_BF))
                accuracy3=np.column_stack((accuracy2,acc_WBF))               
                Re=np.column_stack((movie,accuracy3))
                Res_df=pd.DataFrame(Re)
                #print('   movie_id    ratings            avg_rating   pred_AF   pred_BF    pred_WBF    |avg_rating-pred_AF|     |avg_rating-pred_BF|     |avg_rating-pred_WBF| ')       
                #print(Res_df)
                
                #Res.columns=['movie_id','ratings','avg_rating','pred_AF','pred_BF','pred_WBF','|avg_rating-pred_AF|','|avg_rating-pred_BF|','|avg_rating-pred_WBF|']
                #np.savetxt(r'C:\Users\USER\Desktop\accuracy',Res.values, fmt='%d')
                accur_AF.append(np.mean(acc_AF))
                accur_BF.append(np.mean(acc_BF))
                accur_WBF.append(np.mean(acc_WBF))
                Group.append(N)
                #prceision & recall
                arr=[]
                c=0
                for i in range(0,R_means.shape[0]):
                        if(R_means.item(i)>3):
                                arr.append(i)
                                c=c+1
                s1=np.argsort(-p_AF1)
                s2=np.argsort(-p_BF)
                s3=np.argsort(-p_WBF)
                rec1=[]
                rec2=[]
                rec3=[]
                for i in range(0,20):
                        rec1.append(s1.item(i))
                        rec2.append(s2.item(i))     
                        rec3.append(s3.item(i))
                tp1=0
                tp2=0
                tp3=0
                for i in range(0,20):
                        for j in range(0,len(arr)):
                                if(arr[j]==rec1[i]):
                                        tp1=tp1+1
                                if(arr[j]==rec2[i]):
                                        tp2=tp2+1
                                if(arr[j]==rec3[i]):
                                        tp3=tp3+1
                                
                p1=float(tp1)/20
                r1=float(tp1)/c
                prec1.append(p1)
                recall1.append(r1)
                p2=float(tp2)/20
                r2=float(tp2)/c
                prec2.append(p2)
                recall2.append(r2)
                p3=float(tp3)/20
                r3=float(tp3)/c
                prec3.append(p3)
                recall3.append(r3)
                F1.append((2*p1*r1)/(p1+r1))
                F2.append((2*p2*r2)/(p2+r2))
                F3.append((2*p3*r3)/(p3+r3))
                
                Res_df.drop(Res_df.index,inplace=True)
                G_df.drop(G_df.index, inplace=True)
                R_df.drop(R_df.index, inplace=True)
                R1_df.drop(R_df.index, inplace=True)
                
        mean_accur_AF.append(np.mean(accur_AF))
        mean_accur_BF.append(np.mean(accur_BF))
        mean_accur_WBF.append(np.mean(accur_WBF))
        M_mean_accur_AF=np.matrix(mean_accur_AF)
        M_mean_accur_BF=np.matrix(mean_accur_BF)
        M_mean_accur_WBF=np.matrix(mean_accur_WBF)
        Group_comp.append(N)

        prec_AF.append(np.mean(prec1))
        recal_AF.append(np.mean(recall1))

        prec_BF.append(np.mean(prec2))
        recal_BF.append(np.mean(recall2))
        prec_WBF.append(np.mean(prec3))
        recal_WBF.append(np.mean(recall3))
        F_AF.append(np.mean(F1))
        F_BF.append(np.mean(F2))
        F_WBF.append(np.mean(F3))
                                
M_Group_comp=np.matrix(Group_comp).T        
#print('mean of |avg_rating-pred_AF|')
m1=np.column_stack((np.transpose(Group),np.transpose(accur_AF)))

#print('mean of |avg_rating-pred_BF|')
m2=np.column_stack((np.transpose(Group),np.transpose(accur_BF)))

#print('mean of |avg_rating-pred_WBF|')
m3=np.column_stack((np.transpose(Group),np.transpose(accur_WBF)))


print('    Group    AF      BF      WBF')
m4=np.column_stack((m1,np.transpose(accur_BF)))
m5=np.column_stack((m4,np.transpose(accur_WBF)))
#print(pd.DataFrame(m5))
#np.savetxt(r'C:\Users\USER\Desktop\accuracy',pd.DataFrame(m5).values, fmt='%d')

#print(mean_accur_AF)
#print(mean_accur_BF)
#print(mean_accur_WBF)

min_accur=[]
for i in range(0,N-1):
        m=min(M_mean_accur_AF.item(i),M_mean_accur_BF.item(i),M_mean_accur_WBF.item(i))
        if(M_mean_accur_AF.item(i)==m):
                min_accur.append(1)
        elif(M_mean_accur_BF.item(i)==m):
                min_accur.append(2)
        else:
                min_accur.append(3)


M_min_accur=np.matrix(min_accur)

c1=np.column_stack((np.transpose(M_mean_accur_WBF),np.transpose(M_min_accur)))
c2=np.column_stack((np.transpose(M_mean_accur_BF),(c1)))
c3=np.column_stack((np.transpose(M_mean_accur_AF),(c2)))
print('mean of prediction Error.....')
print('   Group     AF    BF    WBF    result_technique')
Comp=np.column_stack(((M_Group_comp),(c3)))
print(pd.DataFrame(Comp))

max_prec=[]
for i in range(0,N-1):
        m=max(prec_AF[i],prec_BF[i],prec_WBF[i])
        if(prec_AF[i]==m):
                max_prec.append(1)
        elif(prec_BF[i]==m):
                max_prec.append(2)
        else:
                max_prec.append(3)

final_prec=np.column_stack((M_Group_comp,np.column_stack((np.transpose(prec_AF),np.column_stack((np.transpose(prec_BF),np.column_stack((np.transpose(prec_WBF),np.transpose(max_prec)))))))))
print("group    prec_AF    prec_BF   prec=WBF  max_prec")
print(pd.DataFrame(final_prec))

max_recal=[]
for i in range(0,N-1):
        m=max(recal_AF[i],recal_BF[i],recal_WBF[i])
        if(recal_AF[i]==m):
                max_recal.append(1)
        elif(recal_BF[i]==m):
                max_recal.append(2)
        else:
                max_recal.append(3)
final_recal=np.column_stack((M_Group_comp,np.column_stack((np.transpose(recal_AF),np.column_stack((np.transpose(recal_BF),np.column_stack((np.transpose(recal_WBF),np.transpose(max_recal)))))))))
print("group    recal_AF   recal_BF   recal_WBF  max_recal") 
print(pd.DataFrame(final_recal))

max_F=[]
for i in range(0,N-1):
        m=max(F_AF[i],F_BF[i],F_WBF[i])
        if(recal_AF[i]==m):
                max_F.append(1)
        elif(recal_BF[i]==m):
                max_F.append(2)
        else:
                max_F.append(3)
final_F=np.column_stack((M_Group_comp,np.column_stack((np.transpose(F_AF),np.column_stack((np.transpose(F_BF),np.column_stack((np.transpose(F_WBF),np.transpose(max_F)))))))))
print("group    F_AF   F_BF   F_WBF  max_F") 
print(pd.DataFrame(final_F))

#plot a graph-pred_error
x=M_Group_comp
y1=mean_accur_AF
y2=mean_accur_BF
y3=mean_accur_WBF
plt.subplot(3,1,1)
plt.plot(x,y1,label='AF')
plt.plot(x,y2,label='BF')
plt.plot(x,y3,label='WBF')
plt.title('Random group - Pred_Error')
plt.xlabel('Group')
plt.grid(True)
plt.legend()
#plt.show()



#plot graph-prec
x=M_Group_comp
y1=prec_AF
y2=prec_BF
y3=prec_WBF
plt.subplot(3,1,2)
plt.plot(x,y1,label='AF')
plt.plot(x,y2,label='BF')
plt.plot(x,y3,label='WBF')
plt.title('Random group -Precision')
plt.xlabel('Group')
plt.grid(True)
plt.legend()
#plt.show()

#plot graph-recal
x=M_Group_comp
y1=recal_AF
y2=recal_BF
y3=recal_WBF
plt.subplot(3,1,3)
plt.plot(x,y1,label='AF')
plt.plot(x,y2,label='BF')
plt.plot(x,y3,label='WBF')
plt.title('Random group -Recal')
plt.xlabel('Group')
plt.grid(True)
plt.legend()
plt.show()





