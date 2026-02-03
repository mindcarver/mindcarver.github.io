import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"

const Profile: QuartzComponent = ({ cfg }: QuartzComponentProps) => {
  return (
    <div class="profile">
      <div class="profile-header">
        <a href="https://github.com/mindcarver" target="_blank" rel="noopener" class="profile-avatar-link">
          <img
            src="https://avatars.githubusercontent.com/u/32150062?v=4"
            alt="MindCarver"
            class="profile-avatar"
          />
        </a>
        <div class="profile-info">
          <h2 class="profile-name">MindCarver</h2>
          <p class="profile-bio">技术服务于产品</p>
        </div>
      </div>
    </div>
  )
}

Profile.css = `
.profile {
  padding: 1.25rem 1rem;
  border-bottom: 1px solid var(--lightgray);
  margin-bottom: 0;
}

.profile-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.profile-avatar-link {
  display: block;
  transition: opacity 0.2s;
}

.profile-avatar-link:hover {
  opacity: 0.8;
}

.profile-avatar {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  display: block;
}

.profile-info {
  flex: 1;
}

.profile-name {
  font-size: 1rem;
  font-weight: 600;
  margin: 0;
  color: var(--dark);
}

.profile-bio {
  font-size: 0.8rem;
  color: var(--gray);
  margin: 0.2rem 0 0 0;
}

/* Dark mode */
body.dark .profile {
  border-bottom-color: var(--gray);
}

body.dark .profile-name {
  color: var(--light);
}

body.dark .profile-bio {
  color: var(--gray);
}
`

export default (() => Profile) satisfies QuartzComponentConstructor
