const AGENT_STYLES = {
  CONVERSATION_AGENT:        { bg: '#dcfce7', text: '#166534', label: 'Conversation' },
  RAG_AGENT:                 { bg: '#dbeafe', text: '#1e40af', label: 'RAG' },
  WEB_SEARCH_PROCESSOR_AGENT:{ bg: '#ffedd5', text: '#9a3412', label: 'Web Search' },
  BRAIN_TUMOR_AGENT:         { bg: '#fee2e2', text: '#991b1b', label: 'Brain Tumor' },
  CHEST_XRAY_AGENT:          { bg: '#f3e8ff', text: '#6b21a8', label: 'Chest X-ray' },
  SKIN_LESION_AGENT:         { bg: '#ccfbf1', text: '#115e59', label: 'Skin Lesion' },
  HUMAN_VALIDATED:           { bg: '#e0e7ff', text: '#3730a3', label: 'Validated' },
  System:                    { bg: '#f1f5f9', text: '#475569', label: 'System' },
}

export function getAgentStyle(agentName) {
  for (const [key, style] of Object.entries(AGENT_STYLES)) {
    if (agentName && agentName.includes(key)) return style
  }
  return AGENT_STYLES.System
}

export default function AgentBadge({ agent }) {
  const style = getAgentStyle(agent)
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold"
      style={{ background: style.bg, color: style.text }}
    >
      {style.label}
    </span>
  )
}
