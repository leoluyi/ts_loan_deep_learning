library(reshape2)
library(tidyr)
library(data.table)

# Read data ---------------------------------------------------------------

dt <- fread("CA_DATA_20161205.CSV")
names(dt)
ncol(dt) # 54

# Normalization -----------------------------------------------------------

normalize <- function(x) {
  # x = c(1:50, NA)
  x <- as.numeric(x)
  # x_scaled <- x / sqrt(crossprod(na.omit(x)))
  suppressWarnings(x_scaled <- log((x + 1e-5)))
  # invisible(gc())
  x_scaled
}
cols <- names(dt)[5:54]
system.time(
  dt[, (cols) := lapply(.SD, normalize), .SDcols = cols]
)
#'   user  system elapsed 
#' 46.308   0.740  47.093 

# Reshape -----------------------------------------------------------------

## Split response data
rsp <- dt[, .(NUM, RSP_AMT, RSP_FLG)] %>% unique()

## prepare to melt
dt[, c("RSP_AMT", "RSP_FLG") := NULL]
system.time(
  dt2 <- dt %>% melt(id=c("NUM", "YYYYMM"))
) # elapse 3.719
rm(dt); gc()
# dt3 <- dt2 %>% complete(nesting(YYYYMM, variable), fill = 0)
dt2[, var_combined := paste(YYYYMM, variable, sep = "_")]
dt2[, c("YYYYMM", "variable") := NULL]
gc()

## Cast to wide
system.time(
  dt3 <- dt2 %>% spread(key = var_combined, value, fill = 0)
)
# dt3 <- dt2 %>% dcast.data.table(NUM ~ var_combined, value.var = "value", 
#                      fill = 0, drop=FALSE)
rm(dt2); gc()

## Merge response
out <- merge(dt3, rsp, by = "NUM")
# out %>% fwrite("casted_data.csv")
out %>% fwrite("casted_data_norm.csv")

