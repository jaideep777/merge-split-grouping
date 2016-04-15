#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <fstream>
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
	float pg;
	int ng, kg;
};


class ParticleSystem{
	public:
	int N;	// number of particles
	
	vector <int> gid;	// group ids of all particles
	vector <int> par;	// group parent of each particle
	vector <int> sz;	// tree size of each root

	vector <float> ws;	// cohesive tendency of each particle
	vector <float> wa;	// cooperative tendency of each particle
	
	map <int, Group> grps;

	// dynamics
	float mergeRate, splitRate; // splitting rate compared to merging rate  

	public:
	void init(int n){
		N=n;
		gid.resize(N,0);
		par.resize(N,0);
		sz.resize(N,1);
		ws.resize(N,0);
		for (int i=0; i<N; ++i) {
			par[i] = gid[i] = ws[i] = i;
			wa[i] = (i < N/2)? 1:0;
		}
		splitRate = 3;
		mergeRate = 1;
		
		if (splitRate > 1){
			mergeRate = 1/splitRate;
			splitRate = 1;
		}
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
			cout << i->first << " :\t"
				 << i->second.ws << " | ";
			for (int j=0; j<i->second.members.size(); ++j) cout << i->second.members[j] << " ";
			cout << "(" << sz[i->first] << ")\n";
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
		cout << endl << "ws : ";
		for (int i=0; i<N; ++i) cout << ws[i] << " ";
		cout << endl;
	}
		
		
	// calculate group averages
	void calcGrpAverages(){
		for (map <int, Group>::iterator i=grps.begin(); i!=grps.end(); ++i){
			i->second.ws = i->second.ng = i->second.pg = i->second.kg = 0;
			for (int j=0; j<i->second.members.size(); ++j){
				i->second.ws += ws[i->second.members[j]];
				i->second.ng += 1;
				i->second.kg += wa[i->second.members[j]];
			}
			i->second.ws /= i->second.members.size();
			i->second.pg = float(i->second.kg) / i->second.ng;
		}		
	}
	
	// choose random group (returns root)
	int chooseGrp(){
		map <int, Group> ::iterator it = grps.begin();
		int rnd = rand() % grps.size();
		advance(it, rnd);
		return it->first;
	}

	// choose random pair of groups (returns roots)
	vector <int> chooseGrpPair(){
		map <int, Group> ::iterator it = grps.begin();
		int rnd = rand() % grps.size();
		int rnd2 = rand() % (grps.size()-1);
		if (rnd2 == rnd) rnd2 = grps.size()-1;
		int i1 = min(rnd, rnd2);
		int i2 = max(rnd, rnd2);
		
		vector <int> pair(2);
		advance(it, i1);
		pair[0] = it->first;
		advance(it, i2-i1);
		pair[1] = it->first;
		
		return pair;
	}
	
	float calc_r(){
	
	}	
}; 



float runif(){
	return float(rand())/RAND_MAX;
}


int main(){

	ParticleSystem psys;
	psys.init(1000);

	psys.mapGroups();
	psys.calcGrpAverages();

	int p, q;

	cout << "start...\n";
	ofstream fout("ngrps.txt");
	for (int t=0; t<10000; ++t){
		
		// merge a random group pair
		if (psys.grps.size() > 1 && runif() < psys.mergeRate){
			vector <int> pair = psys.chooseGrpPair();
			psys.merge(pair[0], pair[1]);
			psys.mapGroups();
		}
		
		// split a random group
		if (runif() < psys.splitRate){
			int r = psys.chooseGrp();
			float siz = psys.grps[r].members.size();
			float prob_split = siz*siz/(5*5+siz*siz);  // groups of size 5 have 50% chance of splitting
			if (runif() < prob_split){
				psys.split(r);
				psys.mapGroups();
			}
		}		

		// recalculate grp average quantities
		psys.calcGrpAverages();	
		
		// output group size
		fout << psys.grps.size() << endl;
	}

//	psys.printGroupProps();
//	psys.printGroups();
	
	ofstream fout_gsd("gsd.txt");
	for (map <int, Group>::iterator it = psys.grps.begin(); it != psys.grps.end(); ++it ){
		fout_gsd << it->second.members.size() << " ";
	}
	fout << endl;
	
	
}








