
  1. 原始线失败的根因是“入口太薄 + 费用太重”，不是退出错了。
  2. cooldown 只能减亏，救不活。
  3. entry_thicker 证明了高阈值尾部确实有更厚的 alpha。
  4. common traits audit 证明拖后腿的是中间带增量交易。
  5. exclude_high_vol 是把这条线从负拉到正的关键正交过滤。
  6. 当前最干净的正式结论是：entry_z=2.40 + high-vol exclusion(window=16) 已经能把 05 和 06 都拉成正收益。
  7. 但它还不是 GO，因为 holdout 太稀疏，28 个日窗里只有 8 天有交易，所以现在最准确的状态是：已经找到能转正的稀疏候选，但还没到稳定可运营。
