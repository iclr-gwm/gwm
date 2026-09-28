(function () {
  const filters = document.querySelector('.related-filters');
  const buttons = Array.from(document.querySelectorAll('[data-related-filter]'));
  const rows = Array.from(document.querySelectorAll('[data-related-group]'));
  const count = document.getElementById('related-count');
  if (!filters || !rows.length) return;

  function showGroup(group) {
    let visible = 0;
    let previousGroup = 'gwm';
    rows.forEach((row) => {
      const current = row.dataset.relatedGroup;
      row.hidden = group !== 'all' && current !== group && current !== 'gwm';
      row.classList.remove('related-group-start');
      if (!row.hidden) {
        visible += 1;
        if (current !== previousGroup && previousGroup !== 'gwm') row.classList.add('related-group-start');
        previousGroup = current;
      }
    });
    buttons.forEach((button) => {
      const active = button.dataset.relatedFilter === group;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    count.textContent = `${visible} of ${rows.length} method rows · GWM included in every view`;
  }

  buttons.forEach((button) => button.addEventListener('click', () => showGroup(button.dataset.relatedFilter)));
  filters.hidden = false;
  showGroup('memory');
})();
