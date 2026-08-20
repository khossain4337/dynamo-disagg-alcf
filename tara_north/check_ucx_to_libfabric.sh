#!/bin/bash 
#
source /vast/draco/tara/projects/Tara_Deployment/software/miniforge3/bin/activate
conda activate /vast/draco/tara/projects/Tara_Deployment/software/envs/conda_envs/vllm_0.27.1_nixl_1.4.0_python_3.12.12
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")

echo "### 0. which nixl variants are installed ###"
ls -d "$SITE"/nixl_cu*.libs 2>/dev/null

echo ""
echo "### 0b. which one does 'import nixl' ACTUALLY load at runtime? ###"
python3 - <<'PY'
import sys, nixl
print("nixl package __file__:", nixl.__file__)
try:
  from nixl._api import nixl_agent  # force the bindings to load
except Exception as e:
  print("import nixl._api raised:", repr(e))
for name in sorted(sys.modules):
  if name.startswith("nixl_cu"):
      print("loaded compiled module:", name)
PY

# ---- target cu13 explicitly from here ----
CU13_LIBS="$SITE/nixl_cu13.libs"
UCX13="$CU13_LIBS/ucx"

echo ""
echo "### 1. cu13 UCX transport modules ###"
ls -1 "$UCX13" 2>/dev/null

echo ""
echo "### 2. cu13: string-scan for fabric/CXI keywords ###"
for pat in cxi cassini slingshot libfabric ugni; do
  echo "-- '$pat' --"
  for so in "$CU13_LIBS"/*.so* "$UCX13"/*.so*; do
      [ -f "$so" ] || continue
      strings -a "$so" 2>/dev/null | grep -qi "$pat" && echo "   HIT: $(basename "$so")"
  done
done

SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
LIBF="$SITE/nixl_cu13.libs/nixl/libplugin_LIBFABRIC.so"

echo "### full NIXL_* env vars in the LIBFABRIC plugin ###"
strings -a "$LIBF" | grep -oE 'NIXL_[A-Za-z0-9_]+' | LC_ALL=C sort -u

echo ""
echo "### topology / rail / discover strings ###"
strings -a "$LIBF" | grep -iE 'topolog|rail|discover' | LC_ALL=C sort -u

echo ""
echo "### fallback / default-policy / skip / disable / proceed ###"
strings -a "$LIBF" | grep -iE 'fallback|default|skip|disable|proceed' | LC_ALL=C sort -u

#echo ""
#LIBF=$(find "$SITE"/nixl_cu13* -name "libplugin_LIBFABRIC.so" 2>/dev/null | head -1)
#echo "### 3. cu13 LIBFABRIC plugin: ${LIBF:-<not found>} ###"
#echo "-- NIXL_* env knobs baked in --"
#strings -a "$LIBF" 2>/dev/null | grep -oE 'NIXL_[A-Z0-9_]+' | sort -u
#echo "-- topology / skip / disable / fallback / bypass --"
#strings -a "$LIBF" 2>/dev/null | grep -iE 'topolog|skip|disable|fallback|bypass|force|no.?topo' | sort -u#
