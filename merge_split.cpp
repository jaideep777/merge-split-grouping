#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
using namespace std;


// given array of parents, find the root of q
int root(int q, int* par){
	while (q != par[q]){
		par[q] = par[par[q]];
		q = par[q];
	}
	return q;
}


// check if p and q have same root, i.e. belong to same group
bool find(int p, int q, int *par){
	return root(p, par) == root(q, par);
}


// put p and q in same group by merging the smaller tree into large one
void unite(int p, int q, int *par, int *sz){
	int i = root(p, par);
	int j = root(q, par);
	if (i==j) return;	// if both already have the same root do nothing
	if (sz[i] < sz[j]) {par[i]=j; sz[j] += sz[i];}
	else 			   {par[j]=i; sz[i] += sz[j];}
}


// split a group with root r 
void splitGrp(vector <int> &grp, int * par, int * sz){
//	cout << "split " << r << ", size = " << grpsiz << endl;

	int grpsiz = grp.size();
	if (grpsiz < 2) return;

	// create 2 new roots
	int r1 = grp[0];
	int r2 = grp[1];
	
	sz[r1] = sz[r2] = 1;
	par[r1] = r1;
	par[r2] = r2;
	
	if (grpsiz < 3) return;  // only 2 members in group, so split complete 
	
	for (int i=2; i<grpsiz; ++i){
//		cout << "splitting..\n";
		bool I_grp1 = float(rand())/RAND_MAX < 0.5;
		if (I_grp1){
			++sz[r1];
			par[grp[i]] = r1;
		}
		else{
			++sz[r2];
			par[grp[i]] = r2;
		}
	}
}


class Group{
	public:
	vector <int> members;
	float ws;
};


class ParticleSystem{
	public:
	int N;	// number of particles
	
	vector <int> gid;	// group ids of all particles
	vector <int> par;	// group parent of each particle
	vector <int> sz;	// tree size of each root

	vector <int> ws;	// cohesive tendency of each particle
	
	map <int, Group> grps;

	void init(int n){
		N=n;
		gid.resize(N,0);
		par.resize(N,0);
		sz.resize(N,1);
		ws.resize(N,0);
		for (int i=0; i<N; ++i) par[i] = gid[i] = i;
	}
	
	// create group_id -> member_list map from gids
	void mapGroups(){
		grps.clear();
		for (int i=0; i<N; ++i) {
			int r = root(i, &par[0]);	  // get grp id = root of i
			gid[i] = r;
			grps[r].members.push_back(i); // add i as member to grp r
		}
	}

	// print groups in a single line
	void printGroups(){
		for (map <int, Group>::iterator i=grps.begin(); i!=grps.end(); ++i){
			for (int j=0; j<i->second.members.size(); ++j) cout << i->second.members[j] << " ";
			cout << " | ";
		}		
		cout << endl; 	
	}

	// print groups along wth group properties
	void printGroupProps(){
		for (map <int, Group>::iterator i=grps.begin(); i!=grps.end(); ++i){
			cout << sz[i->first] << ": ";
			for (int j=0; j<i->second.members.size(); ++j) cout << i->second.members[j] << " ";
			cout << "\n";
		}		
		cout << endl; 	
	}


	// unite wrapper: merge groups containing i and j
	void merge(int i, int j){
		unite(i,j, &par[0], &sz[0]); 
	} 


	// split wrapper: split grp rootp
	void split(int rootp){
		splitGrp(grps[rootp].members, &par[0], &sz[0]);
	}


	// print internal vectors
	void printInternals(){
		cout << "id : ";
		for (int i=0; i<N; ++i) cout << i << " ";
		cout << endl << "par: ";
		for (int i=0; i<N; ++i) cout << par[i] << " ";
		cout << endl << "sz : ";
		for (int i=0; i<N; ++i) cout << sz[i] << " ";
		cout << endl << "gid: ";
		for (int i=0; i<N; ++i) cout << gid[i] << " ";
		cout << endl;
	}
		
}; 






int main(){

	ParticleSystem psys;
	psys.init(10);

	psys.merge(0,1);
	psys.merge(0,2);
	psys.merge(0,3);

	psys.merge(5,6);
	psys.merge(5,7);

	psys.merge(8,9);

	psys.merge(0,5);
	psys.merge(0,9);


	psys.mapGroups();
	psys.printGroupProps();
	psys.printInternals();


	int p, q;
	while (1){
		cout << ">> ";
		string s;
		cin >> s;

		if (s == "m") {
			cin >> p >> q;
			psys.merge(p,q);
			psys.mapGroups();
		}	
		else{
			cin >> p;
			int rootp = root(p, &psys.par[0]);
			psys.split(rootp);
			psys.mapGroups();
		}
	
		psys.printGroupProps();
		psys.printGroups();
		psys.printInternals(); 
	}
	
}






// remove p from its group and put it in separate group
//void remove(int p, int *par, int *sz, int*grp, int grpsiz){
//	int r = root(p, par);
//	if (p != r){
//		--sz[r];
//		sz[p] = 1;
//		// set parent of everyone in the grp as root
//		for (int i=0; i<grpsiz; ++i){
//			par[grp[i]] = r;
//		}
//		par[p] = p;
//	}
//	else {	// element to be removed is root
//		int newroot;
//		if (grp[0] == r){ // if 1st element is root make 2nd as root
//			newroot = grp[1];
//		}
//		else{	// 1st element is not root, so it can be new root
//			newroot = grp[0];
//		}		
//		// set parent of everyone to new root
//		for (int i=1; i<grpsiz; ++i){
//			par[grp[i]] = newroot;
//		}
//		par[p] = p;	// set parent of p = p, so p becomes root
//		sz[newroot] = sz[p]-1;
//		sz[p] = 1;
//	}
//}


