import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"

const Profile: QuartzComponent = ({ cfg }: QuartzComponentProps) => {
  return (
    <div class="profile">
      <div class="profile-header">
        <img
          src="https://avatars.githubusercontent.com/u/32150062?v=4"
          alt="MindCarver"
          class="profile-avatar"
        />
        <div class="profile-info">
          <h2 class="profile-name">MindCarver</h2>
          <p class="profile-bio">技术服务于产品</p>
        </div>
      </div>

      <div class="profile-description">
        <p>深山中的匠心，专注于区块链开发与量化研究。相信最深刻的创新来自专注与坚持。</p>
      </div>

      <div class="profile-skills">
        <div class="skill-group">
          <span class="skill-icon">🔗</span>
          <div>
            <strong>区块链</strong>
            <p class="skill-detail">智能合约 · DeFi · ZK证明</p>
          </div>
        </div>
        <div class="skill-group">
          <span class="skill-icon">🤖</span>
          <div>
            <strong>机器学习</strong>
            <p class="skill-detail">PyTorch · 时序预测 · 交易策略</p>
          </div>
        </div>
        <div class="skill-group">
          <span class="skill-icon">📈</span>
          <div>
            <strong>量化交易</strong>
            <p class="skill-detail">统计套利 · 风险管理 · MEV</p>
          </div>
        </div>
      </div>

      <div class="profile-links">
        <a href="https://github.com/mindcarver" target="_blank" rel="noopener" class="profile-link">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
          </svg>
          GitHub
        </a>
      </div>
    </div>
  )
}

Profile.css = `
.profile {
  padding: 1.5rem 1rem;
  border-bottom: 1px solid var(--lightgray);
  margin-bottom: 0;
}

.profile-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1rem;
}

.profile-avatar {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  border: 2px solid var(--secondary);
}

.profile-info {
  flex: 1;
}

.profile-name {
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0;
  color: var(--dark);
}

.profile-bio {
  font-size: 0.85rem;
  color: var(--gray);
  margin: 0.25rem 0 0 0;
}

.profile-description {
  font-size: 0.9rem;
  color: var(--darkgray);
  line-height: 1.6;
  margin-bottom: 1rem;
  padding: 0.75rem;
  background: var(--lightgray);
  border-radius: 8px;
}

.profile-skills {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-bottom: 1rem;
}

.skill-group {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  font-size: 0.85rem;
}

.skill-icon {
  font-size: 1.2rem;
  flex-shrink: 0;
}

.skill-group strong {
  display: block;
  color: var(--dark);
}

.skill-detail {
  margin: 0;
  color: var(--gray);
  font-size: 0.8rem;
}

.profile-links {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.profile-link {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.4rem 0.75rem;
  background: var(--secondary);
  color: white !important;
  text-decoration: none;
  border-radius: 20px;
  font-size: 0.85rem;
  transition: opacity 0.2s;
}

.profile-link:hover {
  opacity: 0.85;
}

/* Dark mode adjustments */
body.dark .profile {
  border-bottom-color: var(--gray);
}

body.dark .profile-description {
  background: var(--darkgray);
  color: var(--lightgray);
}

body.dark .profile-name,
body.dark .skill-group strong {
  color: var(--light);
}

body.dark .profile-bio,
body.dark .skill-detail {
  color: var(--gray);
}
`

export default (() => Profile) satisfies QuartzComponentConstructor
