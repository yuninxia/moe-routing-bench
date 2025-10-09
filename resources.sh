#!/usr/bin/env bash

NEED="${NEED:-4}"
GPU="${GPU:-lovelace}"
PARTS="${PARTS:-long commons scavenge debug}"

for P in $PARTS; do
  echo "$P (MaxTime=$(scontrol show partition "$P" 2>/dev/null | awk -F'[= ]' '/MaxTime=/{print $2; exit}')):"
  sinfo -N -p "$P" -o "%20N %10T %30G %C" | grep -i "$GPU" || true

  # --- Summary: node count / nodes satisfying >=NEED / total free GPUs / names of nodes ready to satisfy request ---
  sinfo -h -N -p "$P" -o "%N|%G|%C" | awk -F'|' -v need="$NEED" -v arch="$GPU" '
    function tolower_str(s){ for(i=1;i<=length(s);i++){c=substr(s,i,1); if(c>="A" && c<="Z") s=substr(s,1,i-1) "" tolower(c) "" substr(s,i+1); } return s }
    function used_from_S(s,   n,i,a,b,lo,hi){ n=0; if(s=="") return 0;
      split(s,a,","); for(i in a){
        if (a[i] ~ /^[0-9]+-[0-9]+$/){ split(a[i],b,"-"); lo=b[1]; hi=b[2]; n+=hi-lo+1 }
        else if (a[i] ~ /^[0-9]+$/){ n+=1 }
      } return n
    }
    {
      g=tolower_str($2)
      if (index(g,"gpu:" tolower_str(arch))==0) next
      # Extract total GPU count
      total = 0
      match($2,/gpu:[^:]*:([0-9]+)/,m); if(m[1]!="") total=m[1]+0
      if(total==0) next
      # Extract S:... (used GPU indices) and calculate used count
      S=""
      match($2,/\(S:([^)]*)\)/,s); if(s[1]!="") S=s[1]
      used = used_from_S(S)
      free = total - used
      # Accumulate counts
      nodes++
      if (free>0) free_sum += free
      if (free>=need){ ok++; list = list $1 " " }
    }
    END{
      if(nodes==0) { print "summary: nodes=0  ready(>=" need ")=0  free_gpus=0  ready_nodes=-"; exit }
      print "summary: nodes=" nodes "  ready(>=" need ")=" ok "  free_gpus=" free_sum "  ready_nodes=" (ok?list:"-")
    }'
  echo
done
