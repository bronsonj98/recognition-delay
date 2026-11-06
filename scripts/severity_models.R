## Fine-Gray models of recognition: identical participants and covariates, fitted
## without and with enrollment-severity splines. Called by compare_groups.py.
suppressMessages(library(survival))
suppressMessages(library(splines))
args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(file("stdin"))
d$sex <- factor(ifelse(d$sex == 0, "women", "men"), levels=c("men", "women"))
d$tq <- factor(d$tq, levels=0:4, labels=paste0("Q", 1:5))
d$provider <- factor(d$provider_id)
d$age10 <- d$age/10
d$year <- d$year-2008
out <- NULL
for (cond in unique(d$condition)) {
  x <- d[d$condition == cond,]
  cols <- c("t","e","id","age10","sex","tq","nonwhite","provider","tested","year","severity")
  if (cond == "ckd_confirmed") cols <- c(cols,"cystatin_egfr")
  x <- x[complete.cases(x[,cols]),cols]
  x$status <- factor(x$e, levels=0:2, labels=c("censored","recognized","death"))
  base <- "age10 + sex + tq + nonwhite + provider + tested + year"
  full <- paste(base, "+ ns(severity,df=3)")
  if (cond == "ckd_confirmed") full <- paste(full,"+ ns(cystatin_egfr,df=3)")
  for (adj in c("unadjusted","severity")) {
    rhs <- if (adj == "unadjusted") base else full
    fg <- finegray(as.formula(paste("Surv(t,status) ~",rhs,"+ id")), data=x, etype="recognized")
    covcols <- setdiff(names(fg),c("id","fgstart","fgstop","fgstatus","fgwt"))
    names(fg)[match(covcols,names(fg))] <- paste0("v",seq_along(covcols))
    fgrhs <- paste(paste0("v",seq_along(covcols)),collapse=" + ")
    fgfit <- coxph(as.formula(paste("Surv(fgstart,fgstop,fgstatus) ~",fgrhs,"+ cluster(id)")),
                   weights=fgwt,data=fg)
    for (tag in c("fine_gray")) {
      s <- summary(fgfit)$coefficients
      secol <- if ("robust se" %in% colnames(s)) "robust se" else "se(coef)"
      terms <- rownames(s)
      if (tag=="fine_gray") {
        for (j in rev(seq_along(covcols))) {
          terms <- sub(paste0("^v",j,"(?=[^0-9]|$)"),covcols[j],terms,perl=TRUE)
        }
      }
      a <- data.frame(condition=cond,model=tag,adjustment=adj,term=terms,
                      hr=exp(s[,"coef"]),lo=exp(s[,"coef"]-1.96*s[,secol]),
                      hi=exp(s[,"coef"]+1.96*s[,secol]),p=s[,"Pr(>|z|)"],
                      n=nrow(x),events=sum(x$e==1))
      # Holm over the prespecified group contrasts only (the quantities reported).
      pre <- a$term %in% c("sexwomen","tqQ5","nonwhite","tested")
      a$p_holm <- NA
      a$p_holm[pre] <- p.adjust(a$p[pre],method="holm")
      out <- rbind(out,a)
    }
  }
  cat(cond,nrow(x),"participants;",sum(x$e==1),"recognitions\n")
}
write.table(out,file.path(args[1],"severity_decomposition.tsv"),sep="\t",row.names=FALSE,quote=FALSE)
