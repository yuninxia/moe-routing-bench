#!/usr/bin/env bash
# NOTS/SLURM quick view for L40S GPUs, with precise free-GPU summary.
# Requirements: bash, sinfo, scontrol, awk (GNU date if SHOW_UPTIME=1)

# --- Tunables via env vars ---
NEED="${NEED:-4}"                           # GPUs per node you require (e.g., 1 or 4)
GPU="${GPU:-lovelace}"                      # GPU arch token in GRES (L40S -> "lovelace")
PARTS="${PARTS:-long commons scavenge debug}"  # partitions to inspect
CPT="${CPT:-0}"                             # optional min Idle CPUs per node (0 = ignore)
SHOW_UPTIME="${SHOW_UPTIME:-0}"             # 1 to print uptime table (off by default)

for P in $PARTS; do
  # MaxTime for this partition
  MAXTIME=$(scontrol show partition "$P" 2>/dev/null \
    | awk '{for(i=1;i<=NF;i++) if($i ~ /^MaxTime=/){split($i,a,"="); print a[2]; exit}}')
  echo "$P (MaxTime=${MAXTIME:-?}):"

  # Raw per-node view (quick eyeball)
  sinfo -N -p "$P" -o "%20N %10T %30G %C" | grep -i "gpu:$GPU" || true

  # Get ALL node names in this partition (don’t pre-filter by GRES here)
  NODELIST=$(sinfo -h -N -p "$P" -o "%N" | tr '\n' ' ')
  if [[ -n "$NODELIST" ]]; then
    # Precise summary: prefer TRES (gres/gpu & gres/gpu:<arch>); fallback to Gres/GresUsed.
    # Also (optionally) require Idle CPUs >= CPT.
    scontrol -o show node $NODELIST 2>/dev/null \
    | awk -v need="$NEED" -v arch="$GPU" -v cpt="$CPT" -v cpu_file="/dev/fd/3" '
      BEGIN{
        # Build node -> IdleCPU map from sinfo (fed via FD 3)
        while ((getline line < cpu_file) > 0) {
          split(line, parts, "|")
          split(parts[2], c, "/")     # "alloc/idle/other/total"
          cpu_idle[parts[1]] = c[2] + 0
        }
        close(cpu_file)
      }
      # Prefer TRES key "gres/gpu:<arch>", else "gres/gpu"
      function tres_pick(s, base, arch,   i,a,k,v,best){
        split(s,a,","); best=0
        for(i=1;i<=length(a);i++){
          split(a[i],kv,"="); k=kv[1]; v=kv[2]+0
          if(k==(base ":" arch)) return v
          if(k==base) best=v
        }
        return best
      }
      # Fallback for Gres/GresUsed: prefer "gpu:<arch>:N", else "gpu:N"
      function gres_pick(s, arch,   m){
        if (match(s, "gpu:" arch ":([0-9]+)", m)) return m[1]+0
        else if (match(s, "gpu:([0-9]+)", m))     return m[1]+0
        return 0
      }
      {
        node=""; cfg=""; alloc=""; gres=""; gresused=""; state=""
        # scontrol -o emits key=value tokens per node
        for(i=1;i<=NF;i++){
          split($i,kv,"=")
          if(kv[1]=="NodeName")    node=kv[2]
          else if(kv[1]=="CfgTRES")   cfg=kv[2]
          else if(kv[1]=="AllocTRES") alloc=kv[2]
          else if(kv[1]=="Gres")      gres=kv[2]
          else if(kv[1]=="GresUsed")  gresused=kv[2]
          else if(kv[1]=="State")     state=kv[2]
        }

        # Does this node actually have the target GPU arch?
        has_total_tres = tres_pick(cfg,   "gres/gpu", arch)
        has_total_gres = gres_pick(gres,  arch)
        total = (has_total_tres>0) ? has_total_tres : has_total_gres
        if (total==0) next  # skip nodes without the target GPU

        used_tres  = tres_pick(alloc, "gres/gpu", arch)
        used_gres  = (length(gresused) ? gres_pick(gresused, arch) : 0)
        used = (used_tres>0) ? used_tres : used_gres
        free = total - used

        nodes++
        if (free>0)  free_sum += free
        idle = (node in cpu_idle) ? cpu_idle[node] : 0
        if (free>=need && (cpt<=0 || idle>=cpt)) { ok++; list = list node " " }
      }
      END{
        if(!nodes){
          print "summary: nodes=0  ready(>=" need ")=0  free_gpus=0  ready_nodes=-"
          exit
        }
        print "summary: nodes=" nodes "  ready(>=" need ")=" ok \
              "  free_gpus=" free_sum "  ready_nodes=" (ok?list:"-")
      }' 3< <(sinfo -h -N -p "$P" -o "%N|%C")
  else
    echo "summary: nodes=0  ready(>=$NEED)=0  free_gpus=0  ready_nodes=-"
  fi

  echo

  # Optional uptime table (off by default): SHOW_UPTIME=1
  if [[ "$SHOW_UPTIME" -eq 1 && -n "$NODELIST" ]]; then
    NOW_EPOCH=$(date +%s)
    echo "uptime ($P):"
    scontrol -o show node $NODELIST 2>/dev/null \
    | awk -v now="$NOW_EPOCH" '
      function to_epoch(ts, cmd,ret){ if(ts=="")return -1; cmd="date -d \"" ts "\" +%s"; cmd|getline ret; close(cmd); return ret+0 }
      function fmt_dh(sec, d,h){ if(sec<0)return "?"; d=int(sec/86400); h=int((sec%86400)/3600); return d "d" sprintf("%02dh",h) }
      {
        node=""; bt=""; st=""; state=""
        for(i=1;i<=NF;i++){
          split($i,kv,"=")
          if(kv[1]=="NodeName") node=kv[2]
          else if(kv[1]=="BootTime") bt=kv[2]
          else if(kv[1]=="StateTime") st=kv[2]
          else if(kv[1]=="State") state=kv[2]
        }
        bte=to_epoch(bt); ste=to_epoch(st)
        up=(bte>0)?(now-bte):-1
        age=(ste>0)?(now-ste):-1
        printf "  %-10s  boot=%-19s  up=%-8s  state=%-10s  since=%-19s  age=%-8s\n",
               node, bt, fmt_dh(up), state, st, fmt_dh(age)
      }'
    echo
  fi
done
